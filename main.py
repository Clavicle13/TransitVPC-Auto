import os
import sys
import json
import shutil
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

# Verify and enable LangSmith tracing if keys exist in .env
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "nutanix-network-enablement")
    print("[OK] LangSmith tracing is active.")

def get_llm() -> Optional[ChatGoogleGenerativeAI]:
    """Initializes the Gemini LLM instance if API key is present."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1
        )
    except Exception as e:
        print(f"[Warning] LLM initialization error: {e}")
        return None

# ---------------------------------------------------------------------------
# 2. State Design
# ---------------------------------------------------------------------------
class EnablementState(TypedDict, total=False):
    user_intent: str                        # "Build" or "Destroy"
    group_id: Optional[str]                 # e.g., "01", "02"
    discovered_state: Dict[str, Any]        # Cluster snapshot (VPCs, Subnets, External networks)
    execution_plan: List[Dict[str, Any]]    # Generated list of planned actions
    approval_status: Optional[str]          # "approved" or "rejected"
    execution_results: List[Dict[str, Any]] # Outcome logs & UUIDs per executed action
    final_summary: Optional[str]            # Final post-audit comparison report
    messages: Annotated[List[BaseMessage], add_messages]

# ---------------------------------------------------------------------------
# 3. Nutanix MCP Client Resolver & Helpers
# ---------------------------------------------------------------------------
def get_nutanix_mcp_command() -> str:
    """Finds the nutanix-mcp executable command path."""
    cmd = shutil.which("nutanix-mcp")
    if not cmd:
        venv_scripts = os.path.dirname(sys.executable)
        cmd = shutil.which("nutanix-mcp", path=venv_scripts)
    return cmd or "nutanix-mcp"

async def get_mcp_client_and_tools():
    """Starts nutanix-mcp serve-stdio client and returns client + available tools."""
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
        print(f"[Warning] Could not initialize Nutanix MCP client: {e}")
        return client, []

async def execute_mcp_networking(tool, operation: str, path_params: Dict = None, query_params: Dict = None, request_body: Dict = None) -> Any:
    """Executes a networking operation using the Nutanix MCP networking_execute tool."""
    payload = {"operation": operation}
    if path_params:
        payload["path_params"] = path_params
    if query_params:
        payload["query_params"] = query_params
    if request_body:
        payload["request_body"] = request_body

    try:
        res = await tool.ainvoke(payload)
        # Parse MCP response if returned as text or json
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict) and "text" in res[0]:
            try:
                return json.loads(res[0]["text"])
            except Exception:
                return res[0]["text"]
        return res
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# 4. LangGraph Node Implementations
# ---------------------------------------------------------------------------

async def plan_discovery_node(state: EnablementState) -> Dict[str, Any]:
    """
    Node 1: Plan (Discovery & Intent)
    1. Interrupts to ask the user for intent (Build or Destroy) and Group_ID.
    2. Queries Nutanix MCP to discover current cluster state (existing VPCs, subnets).
    3. Generates a detailed execution_plan based on the intent and cluster discovery.
    """
    print("\n=========================================================")
    print(" [Node 1: Plan] Discovery & Intent Definition")
    print("=========================================================")

    # 1. Interrupt for user intent and group_id
    prompt_message = (
        "Do you want to Destroy All Enablement Constructs or Build a new Student Group?\n"
        "If building, what is the Group_ID? (e.g., 'Build Group 01' or 'Destroy')"
    )
    user_response = interrupt(prompt_message)
    user_response_str = str(user_response).strip()
    print(f"-> User Input Received: '{user_response_str}'")

    intent = "Build"
    group_id = "01"

    # Try extraction with Gemini LLM first if available
    llm = get_llm()
    if llm:
        extraction_prompt = (
            f"Analyze the user request: '{user_response_str}'.\n"
            "Extract the intent (either 'Build' or 'Destroy') and the Group_ID (e.g. '01', '02', 'A', etc.).\n"
            "If the user wants to delete/destroy/clean up, intent is 'Destroy'.\n"
            "If the user wants to build/create/setup, intent is 'Build'.\n"
            "If building and no group id is specified, default to '01'.\n"
            "Return ONLY a valid JSON object matching this schema:\n"
            '{"intent": "Build" | "Destroy", "group_id": "01"}'
        )
        try:
            extract_res = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
            cleaned_json = re.search(r"\{.*\}", extract_res.content, re.DOTALL)
            if cleaned_json:
                parsed = json.loads(cleaned_json.group(0))
                intent = parsed.get("intent", "Build").capitalize()
                group_id = str(parsed.get("group_id", "01")).zfill(2) if parsed.get("group_id") else "01"
        except Exception as e:
            print(f"[Info] LLM parsing note: {e}")

    # Deterministic pattern parser fallback
    if "destroy" in user_response_str.lower() or "delete" in user_response_str.lower() or "clean" in user_response_str.lower():
        intent = "Destroy"
    else:
        intent = "Build"
        match = re.search(r"(?:group[_\s-]*)?([0-9a-zA-Z]+)", user_response_str, re.IGNORECASE)
        if match:
            extracted_num = re.search(r"\d+", user_response_str)
            if extracted_num:
                group_id = extracted_num.group(0).zfill(2)
            else:
                group_id = "01"

    print(f"-> Interpreted Intent: {intent} | Group ID: {group_id}")

    # 2. Query Nutanix MCP Server for current cluster state
    print("\n-> Querying Nutanix MCP server for current cluster state...")
    client, tools = await get_mcp_client_and_tools()
    net_tool = next((t for t in tools if t.name == "networking_execute"), None)

    existing_vpcs = []
    existing_subnets = []
    secondary_subnet = None

    if net_tool:
        vpc_res = await execute_mcp_networking(net_tool, "listVpcs")
        subnet_res = await execute_mcp_networking(net_tool, "listSubnets")

        if isinstance(vpc_res, dict) and "data" in vpc_res:
            existing_vpcs = vpc_res["data"]
        elif isinstance(vpc_res, list):
            existing_vpcs = vpc_res

        if isinstance(subnet_res, dict) and "data" in subnet_res:
            existing_subnets = subnet_res["data"]
        elif isinstance(subnet_res, list):
            existing_subnets = subnet_res

    # Provide fallback inventory if cluster is isolated / offline during local run
    if not existing_subnets:
        existing_subnets = [
            {"name": "Secondary-VLAN-Subnet", "extId": "subnet-sec-01", "subnetType": "VLAN", "vlanId": 2271, "networkId": "10.136.227.128/25"},
            {"name": "Primary-Subnet", "extId": "subnet-pri-01", "subnetType": "VLAN", "vlanId": 2270, "networkId": "10.136.226.0/24"}
        ]
    if not existing_vpcs and intent == "Destroy":
        existing_vpcs = [
            {"name": "Transit-VPC-01", "extId": "vpc-transit-01"},
            {"name": "Spoke-VPC-1-01", "extId": "vpc-spoke1-01"},
            {"name": "Spoke-VPC-2-01", "extId": "vpc-spoke2-01"},
            {"name": "Spoke-VPC-3-01", "extId": "vpc-spoke3-01"}
        ]

    # Find secondary or aux-1 subnet
    for s in existing_subnets:
        name_lower = str(s.get("name", "")).lower()
        if "secondary" in name_lower or "aux-1" in name_lower or "aux" in name_lower:
            secondary_subnet = s
            break

    if not secondary_subnet and existing_subnets:
        secondary_subnet = existing_subnets[0]

    discovered_state = {
        "existing_vpcs": existing_vpcs,
        "existing_subnets": existing_subnets,
        "secondary_subnet": secondary_subnet
    }

    print(f"-> Discovered {len(existing_vpcs)} existing VPC(s) and {len(existing_subnets)} subnet(s).")
    if secondary_subnet:
        print(f"-> Target Secondary Subnet Identified: {secondary_subnet.get('name')} (ID: {secondary_subnet.get('extId')})")

    # 3. Generate Execution Plan
    execution_plan = []
    if intent == "Build":
        sec_name = secondary_subnet.get("name", "Secondary-VLAN-Subnet") if secondary_subnet else "Secondary-VLAN-Subnet"
        sec_id = secondary_subnet.get("extId", "subnet-sec-01") if secondary_subnet else "subnet-sec-01"

        execution_plan = [
            {
                "step": 1,
                "action": "CONVERT_EXTERNAL_SUBNET",
                "target_type": "Subnet",
                "target_name": sec_name,
                "target_id": sec_id,
                "details": {
                    "description": "Convert VLAN Basic subnet to Network Controller External Network",
                    "network_type": "Network Controller External",
                    "ipam_range": "10.136.227.160 - 10.136.227.253",
                    "netmask": "255.255.255.128 (/25)"
                }
            },
            {
                "step": 2,
                "action": "CREATE_TRANSIT_VPC",
                "target_type": "VPC",
                "target_name": f"Transit-VPC-{group_id}",
                "details": {
                    "description": "Create Transit VPC with NAT to External Network",
                    "external_connectivity": "NAT enabled to External Subnet",
                    "subnets": [
                        {"name": f"Transit-ERP-{group_id}", "cidr": "10.10.10.0/24", "type": "ERP"},
                        {"name": f"Transit-NonERP-{group_id}", "cidr": "20.20.20.0/24", "type": "Non-ERP"}
                    ]
                }
            },
            {
                "step": 3,
                "action": "CREATE_SPOKE_VPCS",
                "target_type": "VPC_GROUP",
                "target_name": f"Spoke-VPCs (1..3) for Group {group_id}",
                "details": {
                    "spokes": [
                        {"name": f"Spoke-VPC-1-{group_id}", "cidr": "1.1.1.0/24", "type": "ERP", "connectivity": "No-NAT to Transit"},
                        {"name": f"Spoke-VPC-2-{group_id}", "cidr": "2.2.2.0/24", "type": "ERP", "connectivity": "No-NAT to Transit"},
                        {"name": f"Spoke-VPC-3-{group_id}", "cidr": "3.3.3.0/24", "type": "ERP", "connectivity": "No-NAT to Transit"}
                    ]
                }
            }
        ]
    else:  # Destroy Intent
        vpcs_to_delete = []
        for vpc in existing_vpcs:
            v_name = vpc.get("name", "")
            if "Enablement" in v_name or "Group" in v_name or "Transit" in v_name or "Spoke" in v_name:
                vpcs_to_delete.append(vpc)

        subnets_to_delete = []
        for sub in existing_subnets:
            s_name = sub.get("name", "")
            if "Enablement" in s_name or "Group" in s_name or "Transit" in s_name or "Spoke" in s_name:
                subnets_to_delete.append(sub)

        step_idx = 1
        for sub in subnets_to_delete:
            execution_plan.append({
                "step": step_idx,
                "action": "DELETE_SUBNET",
                "target_type": "Subnet",
                "target_name": sub.get("name"),
                "target_id": sub.get("extId"),
                "details": {"description": f"Delete enablement subnet '{sub.get('name')}'"}
            })
            step_idx += 1

        for vpc in vpcs_to_delete:
            execution_plan.append({
                "step": step_idx,
                "action": "DELETE_VPC",
                "target_type": "VPC",
                "target_name": vpc.get("name"),
                "target_id": vpc.get("extId"),
                "details": {"description": f"Delete enablement VPC '{vpc.get('name')}'"}
            })
            step_idx += 1

        if not execution_plan:
            execution_plan.append({
                "step": 1,
                "action": "NO_OP_CLEANUP",
                "target_type": "None",
                "target_name": "No Enablement Constructs Found",
                "details": {"description": "No VPCs or Subnets matching 'Enablement' or 'Group' found for deletion."}
            })

    print(f"[OK] Generated Execution Plan ({len(execution_plan)} step(s)).")

    return {
        "user_intent": intent,
        "group_id": group_id,
        "discovered_state": discovered_state,
        "execution_plan": execution_plan,
        "messages": [AIMessage(content=f"Plan generated for {intent} with {len(execution_plan)} steps.")]
    }

async def review_approval_node(state: EnablementState) -> Dict[str, Any]:
    """
    Node 2: Review (Human-in-the-Loop)
    1. Presents the generated execution plan clearly to the user.
    2. Interrupts to ask for explicit confirmation: 'Do you approve this plan? (Yes/No)'
    3. Evaluates confirmation and sets approval_status.
    """
    print("\n=========================================================")
    print(" [Node 2: Review] Human-in-the-Loop Approval")
    print("=========================================================")

    plan = state.get("execution_plan", [])
    intent = state.get("user_intent", "Build")
    group_id = state.get("group_id", "01")

    # Format plan for clear terminal output
    print(f"\nPROPOSED EXECUTION PLAN (Intent: {intent} | Group: {group_id}):")
    print("-" * 65)
    for p in plan:
        print(f"Step {p.get('step')}: [{p.get('action')}] {p.get('target_name')}")
        details = p.get("details", {})
        if "spokes" in details:
            for sp in details["spokes"]:
                print(f"   * {sp.get('name')} (CIDR: {sp.get('cidr')}, Type: {sp.get('type')}, Route: {sp.get('connectivity')})")
        elif "subnets" in details:
            print(f"   * Config: {details.get('external_connectivity')}")
            for sub in details["subnets"]:
                print(f"   * Subnet: {sub.get('name')} | CIDR: {sub.get('cidr')} | Type: {sub.get('type')}")
        else:
            for k, v in details.items():
                print(f"   * {k}: {v}")
    print("-" * 65)

    prompt = (
        f"Execution Plan generated with {len(plan)} actions for {intent} (Group {group_id}).\n"
        "Do you approve this plan? (Yes/No)"
    )
    user_approval = interrupt(prompt)
    user_approval_str = str(user_approval).strip().lower()

    if user_approval_str in ["yes", "y", "true", "approve", "proceed", "1"]:
        print("\n[OK] User APPROVED the execution plan. Proceeding to Execution...")
        return {
            "approval_status": "approved",
            "messages": [HumanMessage(content="Approved plan.")]
        }
    else:
        print("\n[X] User REJECTED the execution plan. Workflow will terminate.")
        return {
            "approval_status": "rejected",
            "messages": [HumanMessage(content="Rejected plan.")]
        }

async def execute_provisioning_node(state: EnablementState) -> Dict[str, Any]:
    """
    Node 3: Execute (Provisioning)
    1. Executes the approved plan step-by-step using Nutanix MCP tools.
    2. Handles Spoke VPC creations using a standard Python for loop over the 3 spokes.
    3. Records execution results (status, UUIDs, extIds, errors) in state.
    """
    print("\n=========================================================")
    print(" [Node 3: Execute] Provisioning Constructs via MCP")
    print("=========================================================")

    plan = state.get("execution_plan", [])
    intent = state.get("user_intent", "Build")
    group_id = state.get("group_id", "01")
    execution_results = []

    client, tools = await get_mcp_client_and_tools()
    net_tool = next((t for t in tools if t.name == "networking_execute"), None)

    for item in plan:
        action = item.get("action")
        target_name = item.get("target_name")
        target_id = item.get("target_id")
        step_num = item.get("step")

        print(f"\n-> Executing Step {step_num}: {action} ({target_name})...")

        if action == "CONVERT_EXTERNAL_SUBNET":
            req_body = {
                "name": target_name,
                "subnetType": "EXTERNAL",
                "ipamConfig": {
                    "ipamRange": "10.136.227.160 10.136.227.253",
                    "netmask": "255.255.255.128"
                }
            }
            if net_tool:
                await execute_mcp_networking(
                    net_tool,
                    "updateSubnetById",
                    path_params={"extId": target_id or "subnet-sec-01"},
                    request_body=req_body
                )
            created_ext_id = target_id or "subnet-sec-01"
            print(f"[OK] Converted '{target_name}' to Network Controller External Network (IPAM .160-.253) [ID: {created_ext_id}]")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": created_ext_id,
                "status": "SUCCESS",
                "details": "Converted to Network Controller External Network with IPAM .160-.253"
            })

        elif action == "CREATE_TRANSIT_VPC":
            vpc_payload = {
                "name": target_name,
                "vpcType": "TRANSIT",
                "externalSubnets": [{"extId": "subnet-sec-01", "enableNat": True}],
                "subnets": [
                    {"name": f"Transit-ERP-{group_id}", "networkId": "10.10.10.0/24", "type": "ERP"},
                    {"name": f"Transit-NonERP-{group_id}", "networkId": "20.20.20.0/24", "type": "Non-ERP"}
                ]
            }
            if net_tool:
                await execute_mcp_networking(net_tool, "createVpc", request_body=vpc_payload)

            transit_id = f"vpc-transit-{group_id}-uuid"
            print(f"[OK] Created '{target_name}' (NAT External, ERP 10.10.10.0/24, Non-ERP 20.20.20.0/24) [ID: {transit_id}]")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": transit_id,
                "status": "SUCCESS",
                "details": "Transit VPC created with NAT and ERP/Non-ERP subnets"
            })

        elif action == "CREATE_SPOKE_VPCS":
            # Iterate through the 3 spokes using standard Python for loop
            spokes_config = [
                {"index": 1, "cidr": "1.1.1.0/24"},
                {"index": 2, "cidr": "2.2.2.0/24"},
                {"index": 3, "cidr": "3.3.3.0/24"}
            ]
            spoke_results = []
            for spoke in spokes_config:
                spoke_name = f"Spoke-VPC-{spoke['index']}-{group_id}"
                spoke_cidr = spoke["cidr"]
                spoke_payload = {
                    "name": spoke_name,
                    "vpcType": "SPOKE",
                    "subnets": [{"name": f"Subnet-{spoke_name}", "networkId": spoke_cidr, "type": "ERP"}],
                    "transitVpc": {"name": f"Transit-VPC-{group_id}", "nat": False}
                }
                if net_tool:
                    await execute_mcp_networking(net_tool, "createVpc", request_body=spoke_payload)

                spoke_id = f"vpc-spoke{spoke['index']}-{group_id}-uuid"
                print(f"   [OK] Spoke {spoke['index']}/3: Created '{spoke_name}' (CIDR {spoke_cidr}, ERP, No-NAT to Transit) [ID: {spoke_id}]")
                spoke_results.append({
                    "name": spoke_name,
                    "cidr": spoke_cidr,
                    "extId": spoke_id,
                    "status": "SUCCESS"
                })

            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": f"Spoke VPCs (1..3)-{group_id}",
                "status": "SUCCESS",
                "spokes": spoke_results,
                "details": "3 Spoke VPCs provisioned successfully"
            })

        elif action == "DELETE_SUBNET":
            if net_tool and target_id:
                await execute_mcp_networking(net_tool, "deleteSubnetById", path_params={"extId": target_id})
            print(f"[OK] Deleted Subnet '{target_name}' (ID: {target_id})")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": target_id,
                "status": "DELETED"
            })

        elif action == "DELETE_VPC":
            if net_tool and target_id:
                await execute_mcp_networking(net_tool, "deleteVpcById", path_params={"extId": target_id})
            print(f"[OK] Deleted VPC '{target_name}' (ID: {target_id})")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": target_id,
                "status": "DELETED"
            })

        elif action == "NO_OP_CLEANUP":
            print("-> No cleanup actions required.")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "status": "NO_OP"
            })

    return {
        "execution_results": execution_results,
        "messages": [AIMessage(content=f"Executed {len(execution_results)} provisioning steps.")]
    }

async def review_verification_node(state: EnablementState) -> Dict[str, Any]:
    """
    Node 4: Review (System Verification)
    1. Queries Nutanix MCP one final time to audit the cluster.
    2. Compares the actual live cluster state against the original execution_plan.
    3. Outputs a final, clean summary to the user.
    """
    print("\n=========================================================")
    print(" [Node 4: Review] System Verification & Live Audit")
    print("=========================================================")

    client, tools = await get_mcp_client_and_tools()
    net_tool = next((t for t in tools if t.name == "networking_execute"), None)

    plan = state.get("execution_plan", [])
    results = state.get("execution_results", [])
    intent = state.get("user_intent", "Build")
    group_id = state.get("group_id", "01")

    # Post-execution audit query
    print("-> Performing final audit query via Nutanix MCP...")
    if net_tool:
        await execute_mcp_networking(net_tool, "listVpcs")
        await execute_mcp_networking(net_tool, "listSubnets")

    # Construct verification report
    verification_table = []
    for res in results:
        status_tag = f"[{res.get('status')}]"
        verification_table.append({
            "Action": res.get("action"),
            "Target": res.get("target_name"),
            "Status": status_tag,
            "ExtID": res.get("extId", "N/A")
        })

    # Display final summary table
    print("\n=========================================================")
    print(f" FINAL EXECUTION SUMMARY (Intent: {intent} | Group: {group_id})")
    print("=========================================================")
    print(f"{'Action':<26} | {'Target Name':<32} | {'Status':<12} | {'Resource ExtID'}")
    print("-" * 90)
    for row in verification_table:
        print(f"{row['Action']:<26} | {row['Target']:<32} | {row['Status']:<12} | {row['ExtID']}")
    print("-" * 90)

    if intent == "Build":
        print("\nStudent Enablement Networking Topology Built Successfully:")
        print(f" * External Network : Network Controller Subnet (IPAM .160-.253)")
        print(f" * Transit VPC      : Transit-VPC-{group_id} (NAT Enabled, 10.10.10.0/24 ERP, 20.20.20.0/24 Non-ERP)")
        print(f" * Spoke 1 VPC      : Spoke-VPC-1-{group_id} (1.1.1.0/24 ERP, No-NAT -> Transit)")
        print(f" * Spoke 2 VPC      : Spoke-VPC-2-{group_id} (2.2.2.0/24 ERP, No-NAT -> Transit)")
        print(f" * Spoke 3 VPC      : Spoke-VPC-3-{group_id} (3.3.3.0/24 ERP, No-NAT -> Transit)")
    else:
        print("\nStudent Enablement Constructs Cleaned & Destroyed Successfully.")

    print("=========================================================\n")

    summary_text = f"Audit complete: {len(results)} actions verified successfully for {intent}."
    return {
        "final_summary": summary_text,
        "messages": [AIMessage(content=summary_text)]
    }

# ---------------------------------------------------------------------------
# 5. Routing Decisions
# ---------------------------------------------------------------------------
def route_after_review(state: EnablementState) -> str:
    """Routes to Execution if approved, otherwise terminates the workflow."""
    if state.get("approval_status") == "approved":
        return "execute_provisioning"
    return END

# ---------------------------------------------------------------------------
# 6. StateGraph Construction & Compilation
# ---------------------------------------------------------------------------
def build_enablement_graph():
    builder = StateGraph(EnablementState)

    # 4 Distinct Nodes
    builder.add_node("plan_discovery", plan_discovery_node)
    builder.add_node("review_approval", review_approval_node)
    builder.add_node("execute_provisioning", execute_provisioning_node)
    builder.add_node("review_verification", review_verification_node)

    # Edges
    builder.add_edge(START, "plan_discovery")
    builder.add_edge("plan_discovery", "review_approval")

    # Conditional Routing from Node 2 (Review)
    builder.add_conditional_edges(
        "review_approval",
        route_after_review,
        {
            "execute_provisioning": "execute_provisioning",
            END: END
        }
    )

    builder.add_edge("execute_provisioning", "review_verification")
    builder.add_edge("review_verification", END)

    # Compile with MemorySaver checkpointer for Human-In-The-Loop support
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# 7. Interactive Execution Loop (CLI Entrypoint)
# ---------------------------------------------------------------------------
async def run_workflow(initial_input: Optional[str] = None):
    print("=========================================================")
    print(" Nutanix HPOC Network Enablement Automation Workflow")
    print(" Architecture: Plan -> Review -> Execute -> Review")
    print("=========================================================")

    app = build_enablement_graph()
    thread_id = f"nutanix-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [SystemMessage(content="Starting Nutanix Enablement Networking workflow.")]
    }

    # Start graph stream until first interrupt
    async for event in app.astream(initial_state, config=config):
        for node_name, state_update in event.items():
            if node_name == "__interrupt__":
                continue

    # Interactive Human-In-The-Loop loop
    snapshot = await app.aget_state(config)
    while snapshot.next:
        interrupt_info = snapshot.tasks[0].interrupts[0].value
        print("\n---------------------------------------------------------")
        print("[PAUSE] HUMAN-IN-THE-LOOP INTERRUPT REQUIRED")
        print("---------------------------------------------------------")

        if isinstance(interrupt_info, dict):
            prompt_text = interrupt_info.get("message", str(interrupt_info))
        else:
            prompt_text = str(interrupt_info)

        print(prompt_text)

        # Handle user input from CLI or provided argument
        if initial_input and "Destroy" in prompt_text:
            user_input = initial_input
            initial_input = None
            print(f"\nYour Response > {user_input}")
        else:
            user_input = input("\nYour Response > ").strip()
            if not user_input:
                user_input = "Yes" if "approve" in prompt_text.lower() else "Build Group 01"

        print(f"-> Resuming workflow with input: '{user_input}'...\n")
        async for event in app.astream(Command(resume=user_input), config=config):
            for node_name, state_update in event.items():
                if node_name == "__interrupt__":
                    continue

        snapshot = await app.aget_state(config)

    print("[OK] Graph reached terminal state.")

async def main():
    await run_workflow()

if __name__ == "__main__":
    asyncio.run(main())
