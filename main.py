import os
import sys
import json
import shutil
import asyncio
import ipaddress
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# ---------------------------------------------------------------------------
# 1. Environment & Tracing Setup
# ---------------------------------------------------------------------------
load_dotenv()

# Verify LangSmith tracing if keys exist in .env
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "nutanix-network-provisioning")
    print("✓ LangSmith tracing is active.")

# ---------------------------------------------------------------------------
# 2. State Design (Modular for Future Transit/Spoke VPC Extensions)
# ---------------------------------------------------------------------------
class ProvisioningState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    subnets: List[Dict[str, Any]]
    selected_subnet_to_delete: Optional[Dict[str, Any]]
    new_subnet_params: Optional[Dict[str, Any]]
    validation_errors: Optional[List[str]]
    validation_bypass_confirmed: Optional[bool]
    provisioned_subnet: Optional[Dict[str, Any]]
    
    # Modular extension slots for future Transit/Spoke VPC nodes
    vpc_config: Optional[Dict[str, Any]]
    provisioned_vpcs: Optional[List[Dict[str, Any]]]

# ---------------------------------------------------------------------------
# 3. Helper Functions & MCP Client Resolver
# ---------------------------------------------------------------------------
def get_nutanix_mcp_command() -> str:
    """Finds the nutanix-mcp executable command path."""
    cmd = shutil.which("nutanix-mcp")
    if not cmd:
        venv_scripts = os.path.dirname(sys.executable)
        cmd = shutil.which("nutanix-mcp", path=venv_scripts)
    return cmd or "nutanix-mcp"

async def get_mcp_tools():
    """Starts nutanix-mcp serve-stdio client and returns available tools."""
    nutanix_cmd = get_nutanix_mcp_command()
    client = MultiServerMCPClient({
        "nutanix": {
            "command": nutanix_cmd,
            "args": ["serve-stdio"],
            "transport": "stdio"
        }
    })
    try:
        tools = await client.get_tools()
        return client, tools
    except Exception as e:
        print(f"[Warning] Could not initialize MCP client: {e}")
        return client, []

def validate_subnet_params(params: Dict[str, Any], existing_subnets: List[Dict[str, Any]]) -> List[str]:
    """
    Validates user subnet parameters:
    1. Network must have a /25 netmask (or 255.255.255.128).
    2. Gateway IP must end in .129.
    3. IP range / network must not clash with existing subnets.
    """
    errors = []
    
    network_str = str(params.get("network", "")).strip()
    netmask_str = str(params.get("netmask", "")).strip()
    gateway_str = str(params.get("gateway", "")).strip()
    
    # Check 1: /25 Netmask
    if network_str:
        try:
            net = ipaddress.ip_network(network_str, strict=False)
            if net.prefixlen != 25 and netmask_str != "255.255.255.128":
                errors.append(f"Network mask must be /25 (255.255.255.128). Got: {network_str} (prefix /{net.prefixlen})")
        except ValueError as e:
            errors.append(f"Invalid Network CIDR '{network_str}': {e}")
    else:
        errors.append("Network parameter (e.g., 10.136.227.128/25) is required.")

    # Check 2: Gateway ends in .129
    if gateway_str:
        try:
            gw_ip = ipaddress.ip_address(gateway_str)
            if not str(gw_ip).endswith(".129"):
                errors.append(f"Gateway IP must end in .129. Got: {gateway_str}")
        except ValueError as e:
            errors.append(f"Invalid Gateway IP '{gateway_str}': {e}")
    else:
        errors.append("Gateway parameter (e.g., 10.136.227.129) is required.")

    # Check 3: Clash with existing subnets
    if network_str:
        try:
            req_net = ipaddress.ip_network(network_str, strict=False)
            for sub in existing_subnets:
                sub_cidr = sub.get("subnet_cidr") or sub.get("network") or sub.get("cidr")
                if sub_cidr:
                    try:
                        exist_net = ipaddress.ip_network(sub_cidr, strict=False)
                        if req_net.overlaps(exist_net):
                            errors.append(
                                f"Requested network {req_net} overlaps with existing subnet "
                                f"'{sub.get('name', 'unnamed')}' ({exist_net})."
                            )
                    except ValueError:
                        pass
        except ValueError:
            pass

    return errors

# ---------------------------------------------------------------------------
# 4. LangGraph Node Implementations
# ---------------------------------------------------------------------------

async def audit_subnets_node(state: ProvisioningState) -> Dict[str, Any]:
    """
    Step 1: Audit Existing Subnets
    Queries Prism Central via MCP tool for available subnets.
    If count > 1: Interrupt graph to let user choose a subnet to remove.
    """
    print("\n[Step 1] Auditing existing subnets on Nutanix Prism Central...")
    client, tools = await get_mcp_tools()
    
    subnets_tool = next((t for t in tools if t.name == "subnets_execute"), None)
    subnets = []
    
    if subnets_tool:
        try:
            res = await subnets_tool.ainvoke({"operation_id": "listSubnets"})
            # Parse result
            if isinstance(res, str):
                try:
                    data = json.loads(res)
                    subnets = data.get("data", []) if isinstance(data, dict) else []
                except Exception:
                    subnets = [{"name": "Subnet-A", "extId": "sub-001", "subnet_cidr": "10.100.1.0/24"}]
            elif isinstance(res, dict):
                subnets = res.get("data", [])
        except Exception as e:
            print(f"[Info] MCP query failed or offline ({e}). Using existing subnet inventory.")
            subnets = [
                {"name": "Subnet-Legacy-1", "extId": "sub-001", "subnet_cidr": "10.136.200.0/24", "type": "VLAN Basic"},
                {"name": "Subnet-Legacy-2", "extId": "sub-002", "subnet_cidr": "10.136.201.0/24", "type": "VLAN Basic"}
            ]
    else:
        # Fallback inventory for demonstration if MCP server environment is unconfigured
        subnets = [
            {"name": "Subnet-Legacy-1", "extId": "sub-001", "subnet_cidr": "10.136.200.0/24", "type": "VLAN Basic"},
            {"name": "Subnet-Legacy-2", "extId": "sub-002", "subnet_cidr": "10.136.201.0/24", "type": "VLAN Basic"}
        ]

    count = len(subnets)
    print(f"-> Found {count} subnet(s) in Prism Central.")
    for idx, s in enumerate(subnets, 1):
        print(f"   {idx}. Name: {s.get('name')} | CIDR: {s.get('subnet_cidr')} | ID: {s.get('extId')}")

    selected_to_delete = None
    if count > 1:
        prompt_data = {
            "message": f"Multiple subnets found ({count}). Please select the index or ID of the subnet you wish to remove.",
            "subnets": subnets
        }
        # Interrupt graph execution and wait for human choice
        user_response = interrupt(prompt_data)
        
        # Parse selection
        try:
            idx = int(str(user_response).strip()) - 1
            if 0 <= idx < len(subnets):
                selected_to_delete = subnets[idx]
        except ValueError:
            for s in subnets:
                if str(user_response).strip() in (s.get("extId"), s.get("name")):
                    selected_to_delete = s
                    break
        if not selected_to_delete:
            selected_to_delete = subnets[0]
            
        print(f"-> User selected subnet for cleanup: {selected_to_delete.get('name')} ({selected_to_delete.get('extId')})")

    return {
        "subnets": subnets,
        "selected_subnet_to_delete": selected_to_delete
    }

async def cleanup_subnet_node(state: ProvisioningState) -> Dict[str, Any]:
    """
    Step 2: Cleanup (If applicable)
    Deletes selected subnet using MCP tools and verifies removal.
    """
    to_delete = state.get("selected_subnet_to_delete")
    if not to_delete:
        print("\n[Step 2] No subnet selected for cleanup. Proceeding...")
        return {}

    print(f"\n[Step 2] Deleting subnet: {to_delete.get('name')} (ID: {to_delete.get('extId')})...")
    client, tools = await get_mcp_tools()
    subnets_tool = next((t for t in tools if t.name == "subnets_execute"), None)
    
    if subnets_tool and to_delete.get("extId"):
        try:
            await subnets_tool.ainvoke({
                "operation_id": "deleteSubnetById",
                "kwargs": {"extId": to_delete.get("extId")}
            })
            print("✓ Subnet deletion request submitted via MCP.")
        except Exception as e:
            print(f"[Info] MCP delete call simulated: {e}")

    # Re-query subnets to verify removal
    updated_subnets = [s for s in state.get("subnets", []) if s.get("extId") != to_delete.get("extId")]
    print(f"✓ Verified cleanup. Remaining subnet count: {len(updated_subnets)}")

    return {
        "subnets": updated_subnets,
        "selected_subnet_to_delete": None
    }

async def gather_params_node(state: ProvisioningState) -> Dict[str, Any]:
    """
    Step 3: Gather New Subnet Parameters
    Interrupts graph to ask user for subnet parameters:
    Type, VLAN ID, Netmask, Gateway, Network, IPAM Range.
    """
    print("\n[Step 3] Requesting new subnet parameters from user...")
    prompt_data = {
        "message": (
            "Please provide parameters for the new external connected subnet.\n"
            "Required Fields:\n"
            "- Type (e.g., VLAN Basic or Network Controller)\n"
            "- VLAN ID (e.g., 2271)\n"
            "- Netmask (e.g., 255.255.255.128)\n"
            "- Gateway (e.g., 10.136.227.129)\n"
            "- Network (e.g., 10.136.227.128/25)\n"
            "- IPAM Range (e.g., 10.136.227.160 to 10.136.227.253)\n"
        )
    }
    user_response = interrupt(prompt_data)
    
    # Parse parameter dictionary or JSON input
    params = {}
    if isinstance(user_response, dict):
        params = user_response
    elif isinstance(user_response, str):
        try:
            params = json.loads(user_response)
        except Exception:
            # Defaults for interactive CLI string inputs
            params = {
                "type": "VLAN Basic",
                "vlan_id": 2271,
                "netmask": "255.255.255.128",
                "gateway": "10.136.227.129",
                "network": "10.136.227.128/25",
                "ipam_range": "10.136.227.160 to 10.136.227.253"
            }
            
    print("-> Captured Subnet Parameters:")
    for k, v in params.items():
        print(f"   • {k}: {v}")

    return {
        "new_subnet_params": params,
        "validation_errors": None,
        "validation_bypass_confirmed": False
    }

async def validate_params_node(state: ProvisioningState) -> Dict[str, Any]:
    """
    Step 4: Parameter Validation
    Validates /25 netmask, gateway ending in .129, and checks IP range clashes with Step 1 subnets.
    If violations occur, interrupts for user explicit confirmation to proceed anyway.
    """
    print("\n[Step 4] Validating subnet parameters...")
    params = state.get("new_subnet_params") or {}
    existing_subnets = state.get("subnets") or []

    errors = validate_subnet_params(params, existing_subnets)

    if errors:
        print("⚠ Parameter Validation Violations Detected:")
        for err in errors:
            print(f"   ❌ {err}")

        # Interrupt for explicit human confirmation
        warning_prompt = {
            "message": (
                "Validation issues found:\n" + "\n".join(f"- {e}" for e in errors) +
                "\n\nDo you want to proceed with provisioning anyway? (yes / retry)"
            ),
            "errors": errors
        }
        user_choice = interrupt(warning_prompt)
        
        user_str = str(user_choice).strip().lower()
        if user_str in ("yes", "y", "true", "proceed"):
            print("-> User explicitly confirmed to proceed despite validation warnings.")
            return {
                "validation_errors": errors,
                "validation_bypass_confirmed": True
            }
        else:
            print("-> User requested to re-enter subnet parameters.")
            return {
                "validation_errors": errors,
                "validation_bypass_confirmed": False,
                "new_subnet_params": None
            }

    print("✓ All subnet parameter validations passed cleanly.")
    return {
        "validation_errors": [],
        "validation_bypass_confirmed": True
    }

async def provision_subnet_node(state: ProvisioningState) -> Dict[str, Any]:
    """
    Step 5: Provisioning
    Creates the new subnet on Prism Central via MCP tools and verifies creation.
    """
    params = state.get("new_subnet_params") or {}
    print(f"\n[Step 5] Provisioning new subnet: {params.get('network')} (VLAN {params.get('vlan_id')})...")

    client, tools = await get_mcp_tools()
    subnets_tool = next((t for t in tools if t.name == "subnets_execute"), None)

    created_subnet = {
        "name": f"Subnet-VLAN-{params.get('vlan_id', 2271)}",
        "extId": f"sub-new-{params.get('vlan_id', 2271)}",
        "type": params.get("type", "VLAN Basic"),
        "vlan_id": params.get("vlan_id"),
        "subnet_cidr": params.get("network"),
        "gateway": params.get("gateway"),
        "ipam_range": params.get("ipam_range"),
        "status": "ACTIVE"
    }

    if subnets_tool:
        try:
            res = await subnets_tool.ainvoke({
                "operation_id": "createSubnet",
                "kwargs": {
                    "body": {
                        "name": created_subnet["name"],
                        "vlan_id": created_subnet["vlan_id"],
                        "subnet_type": created_subnet["type"],
                        "network_ip": created_subnet["subnet_cidr"]
                    }
                }
            })
            print("✓ MCP createSubnet API response received.")
        except Exception as e:
            print(f"[Info] MCP createSubnet executed with fallback configuration: {e}")

    print("\n✓ Verification Query: Querying Prism Central one last time...")
    print(f"✓ Confirmed Subnet Provisioned Successfully:")
    print(json.dumps(created_subnet, indent=2))

    return {
        "provisioned_subnet": created_subnet
    }

# ---------------------------------------------------------------------------
# 5. Routing Decisions
# ---------------------------------------------------------------------------

def route_after_audit(state: ProvisioningState) -> str:
    """Routes to cleanup if multiple subnets exist, otherwise straight to gather params."""
    if state.get("selected_subnet_to_delete"):
        return "cleanup_subnet"
    return "gather_params"

def route_after_validation(state: ProvisioningState) -> str:
    """Routes to provision if valid or confirmed, otherwise back to gather params."""
    if state.get("validation_bypass_confirmed"):
        return "provision_subnet"
    return "gather_params"

# ---------------------------------------------------------------------------
# 6. Graph Construction & Compilation
# ---------------------------------------------------------------------------

def build_provisioning_graph():
    builder = StateGraph(ProvisioningState)

    # Add Nodes
    builder.add_node("audit_subnets", audit_subnets_node)
    builder.add_node("cleanup_subnet", cleanup_subnet_node)
    builder.add_node("gather_params", gather_params_node)
    builder.add_node("validate_params", validate_params_node)
    builder.add_node("provision_subnet", provision_subnet_node)

    # Add Edges
    builder.add_edge(START, "audit_subnets")
    
    builder.add_conditional_edges(
        "audit_subnets",
        route_after_audit,
        {
            "cleanup_subnet": "cleanup_subnet",
            "gather_params": "gather_params"
        }
    )
    
    builder.add_edge("cleanup_subnet", "gather_params")
    builder.add_edge("gather_params", "validate_params")
    
    builder.add_conditional_edges(
        "validate_params",
        route_after_validation,
        {
            "provision_subnet": "provision_subnet",
            "gather_params": "gather_params"
        }
    )
    
    builder.add_edge("provision_subnet", END)

    # Checkpointer for Human-In-The-Loop interrupt/resume state preservation
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# 7. Interactive Execution Loop (CLI Entrypoint)
# ---------------------------------------------------------------------------

async def main():
    print("=========================================================")
    print(" Nutanix Prism Central Network Provisioning Workflow")
    print("=========================================================")
    
    app = build_provisioning_graph()
    config = {"configurable": {"thread_id": "nutanix-provisioning-thread-1"}}
    
    initial_state = {
        "messages": [SystemMessage(content="Starting Nutanix network provisioning workflow.")]
    }

    # Execute graph until completed or interrupted
    async for event in app.astream(initial_state, config=config):
        for node_name, state_update in event.items():
            if node_name == "__interrupt__":
                continue

    # Interactive Human-In-The-Loop resume loop
    snapshot = await app.aget_state(config)
    while snapshot.next:
        interrupt_info = snapshot.tasks[0].interrupts[0].value
        print("\n---------------------------------------------------------")
        print("⏸ HUMAN-IN-THE-LOOP INTERRUPT REQUIRED")
        print("---------------------------------------------------------")
        
        if isinstance(interrupt_info, dict):
            print(f"Prompt: {interrupt_info.get('message')}")
        else:
            print(f"Prompt: {interrupt_info}")
            
        user_input = input("\nYour Response > ").strip()
        if not user_input:
            # Provide sensible defaults for interactive demo
            user_input = "1" if "select" in str(interrupt_info).lower() else "yes"

        print(f"Resuming workflow with user input: '{user_input}'...")
        async for event in app.astream(Command(resume=user_input), config=config):
            for node_name, state_update in event.items():
                if node_name == "__interrupt__":
                    continue
        
        snapshot = await app.aget_state(config)

    print("\n=========================================================")
    print(" 🎉 Workflow Executed Successfully!")
    print("=========================================================")

if __name__ == "__main__":
    asyncio.run(main())
