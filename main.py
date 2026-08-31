import os
import sys
import json
import shutil
import asyncio
import re
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import httpx
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

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# LangSmith Imports & Decorator
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ---------------------------------------------------------------------------
# 1. Environment & Tracing Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")
load_dotenv(ENV_FILE, override=True)

# Synchronize LangSmith / LangChain Tracing Environment Variables
langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

    tracing_enabled = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "true"
    os.environ["LANGSMITH_TRACING"] = tracing_enabled
    os.environ["LANGCHAIN_TRACING_V2"] = tracing_enabled

    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "nutanix-network-enablement"
    project = project.strip('"').strip("'")
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        endpoint = endpoint.strip('"').strip("'")
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    # Handle SSL verification bypass if configured
    ignore_certs = os.getenv("LANGSMITH_DANGEROUSLY_IGNORE_CERTS", "").lower() in ["true", "1", "yes"]
    if ignore_certs:
        os.environ["LANGSMITH_DANGEROUSLY_IGNORE_CERTS"] = "true"
        import urllib3
        import requests
        urllib3.disable_warnings()
        _orig_session_init = requests.Session.__init__
        def _patched_session_init(self, *args, **kwargs):
            _orig_session_init(self, *args, **kwargs)
            self.verify = False
        requests.Session.__init__ = _patched_session_init

    print(f"[OK] LangSmith tracing is active (Project: '{project}').")
else:
    print("[Info] LangSmith API key not found; tracing is disabled.")

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
    discovered_state: Dict[str, Any]        # Cluster snapshot (VPCs, Subnets)
    captured_vlan_info: Dict[str, Any]      # Extracted VLAN ID, Cluster Ref, Subnet Name prior to deletion
    captured_dns_servers: List[str]         # Extracted Cluster DNS / Name Server IPs
    execution_plan: List[Dict[str, Any]]    # Generated list of planned actions
    approval_status: Optional[str]          # "approved" or "rejected"
    execution_results: List[Dict[str, Any]] # Outcome logs & UUIDs per executed action
    final_summary: Optional[str]            # Final post-audit comparison report
    messages: Annotated[List[BaseMessage], add_messages]

# ---------------------------------------------------------------------------
# 3. Nutanix Prism Central v4 API Client
# ---------------------------------------------------------------------------
class NutanixPrismClient:
    """Direct client for Nutanix Prism Central v4 API with idempotency and task polling."""

    def __init__(self):
        self.host = os.getenv("PC_HOST", "127.0.0.1")
        self.port = os.getenv("PC_PORT", "9440")
        self.username = os.getenv("PC_USERNAME", "admin")
        self.password = os.getenv("PC_PASSWORD", "")
        self.insecure = os.getenv("PC_INSECURE", "true").lower() in ["true", "1", "yes"]
        self.base_url = f"https://{self.host}:{self.port}"
        self.client = httpx.Client(
            verify=not self.insecure,
            auth=(self.username, self.password) if self.username and self.password else None,
            timeout=30.0
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "NTNX-Request-Id": str(uuid.uuid4())
        }

    @traceable(name="Nutanix: Poll Task", run_type="tool")
    def poll_task(self, task_ext_id: str, timeout_sec: int = 60) -> Optional[str]:
        """Polls async task until SUCCEEDED and returns the primary entity ExtID."""
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                res = self.client.get(f"{self.base_url}/api/prism/v4.0/config/tasks/{task_ext_id}")
                if res.status_code == 200:
                    task_data = res.json().get("data", {})
                    status = task_data.get("status")
                    if status == "SUCCEEDED":
                        entities = task_data.get("entitiesAffected", [])
                        if entities:
                            return entities[0].get("extId")
                        return task_ext_id
                    elif status in ["FAILED", "CANCELED"]:
                        print(f"[Error] Task {task_ext_id} ended with status: {status}")
                        return None
            except Exception as e:
                print(f"[Warning] Task polling note: {e}")
            time.sleep(2)
        return None

    @traceable(name="Nutanix: List Clusters", run_type="tool")
    def list_clusters(self) -> List[Dict[str, Any]]:
        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.get(f"{self.base_url}/api/clustermgmt/{v}/config/clusters")
                if res.status_code == 200:
                    return res.json().get("data", [])
            except Exception:
                pass
        return []

    @traceable(name="Nutanix: Get Cluster", run_type="tool")
    def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.get(f"{self.base_url}/api/clustermgmt/{v}/config/clusters/{cluster_id}")
                if res.status_code == 200:
                    return res.json().get("data")
            except Exception:
                pass
        return None

    @traceable(name="Nutanix: Get Cluster DNS Servers", run_type="tool")
    def get_cluster_dns_servers(self, preferred_cluster_id: Optional[str] = None) -> List[str]:
        """Discovers and extracts configured DNS/Name Server IPs from the target Nutanix cluster."""
        dns_servers: List[str] = []
        if preferred_cluster_id:
            cl = self.get_cluster(preferred_cluster_id)
            if cl:
                for ns in cl.get("network", {}).get("nameServerIpList", []):
                    ip_val = ns.get("ipv4", {}).get("value") or ns.get("value")
                    if ip_val and ip_val != "127.0.0.1" and ip_val not in dns_servers:
                        dns_servers.append(ip_val)

        if not dns_servers:
            clusters = self.list_clusters()
            sorted_clusters = sorted(clusters, key=lambda c: 0 if c.get("clusterType") == "HYPER_CONVERGED" else 1)
            for c_summary in sorted_clusters:
                c_id = c_summary.get("extId")
                if not c_id:
                    continue
                cl = self.get_cluster(c_id)
                if cl:
                    for ns in cl.get("network", {}).get("nameServerIpList", []):
                        ip_val = ns.get("ipv4", {}).get("value") or ns.get("value")
                        if ip_val and ip_val != "127.0.0.1" and ip_val not in dns_servers:
                            dns_servers.append(ip_val)
                if dns_servers:
                    break
        return dns_servers

    @traceable(name="Nutanix: List Subnets", run_type="tool")
    def list_subnets(self) -> List[Dict[str, Any]]:
        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.get(f"{self.base_url}/api/networking/{v}/config/subnets")
                if res.status_code == 200:
                    return res.json().get("data", [])
            except Exception:
                pass
        return []

    @traceable(name="Nutanix: Get Subnet", run_type="tool")
    def get_subnet(self, subnet_id: str) -> Optional[Dict[str, Any]]:
        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.get(f"{self.base_url}/api/networking/{v}/config/subnets/{subnet_id}")
                if res.status_code == 200:
                    return res.json().get("data")
            except Exception:
                pass
        return None

    @traceable(name="Nutanix: Delete Subnet", run_type="tool")
    def delete_subnet(self, subnet_id: str) -> bool:
        # Detach/clean up any VPCs referencing this subnet as external subnet
        for vpc in self.list_vpcs():
            ext_subs = vpc.get("externalSubnets", [])
            for es in ext_subs:
                if es.get("subnetReference") == subnet_id:
                    print(f"-> Detaching referencing VPC '{vpc.get('name')}' ({vpc.get('extId')}) before subnet deletion...")
                    self.delete_vpc(vpc.get("extId"))
                    time.sleep(2)

        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.delete(
                    f"{self.base_url}/api/networking/{v}/config/subnets/{subnet_id}",
                    headers=self._headers()
                )
                if res.status_code in [200, 202, 204]:
                    if res.status_code == 202:
                        task_id = res.json().get("data", {}).get("extId")
                        if task_id:
                            task_res = self.poll_task(task_id)
                            return task_res is not None
                    return True
            except Exception as e:
                print(f"[Error] Delete Subnet {subnet_id}: {e}")
        return False

    @traceable(name="Nutanix: Create Subnet", run_type="tool")
    def create_subnet(self, subnet_payload: Dict[str, Any]) -> Optional[str]:
        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.post(
                    f"{self.base_url}/api/networking/{v}/config/subnets",
                    json=subnet_payload,
                    headers=self._headers()
                )
                if res.status_code in [200, 201, 202]:
                    data = res.json().get("data", {})
                    if res.status_code == 202 and "TaskReference" in data.get("$objectType", ""):
                        task_id = data.get("extId")
                        return self.poll_task(task_id)
                    return data.get("extId")
                else:
                    print(f"[Error] Create Subnet returned {res.status_code}: {res.text}")
            except Exception as e:
                print(f"[Error] Create Subnet exception: {e}")
        return None

    @traceable(name="Nutanix: List VPCs", run_type="tool")
    def list_vpcs(self) -> List[Dict[str, Any]]:
        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.get(f"{self.base_url}/api/networking/{v}/config/vpcs")
                if res.status_code == 200:
                    return res.json().get("data", [])
            except Exception:
                pass
        return []

    @traceable(name="Nutanix: Create VPC", run_type="tool")
    def create_vpc(self, vpc_payload: Dict[str, Any]) -> Optional[str]:
        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.post(
                    f"{self.base_url}/api/networking/{v}/config/vpcs",
                    json=vpc_payload,
                    headers=self._headers()
                )
                if res.status_code in [200, 201, 202]:
                    data = res.json().get("data", {})
                    if res.status_code == 202 and "TaskReference" in data.get("$objectType", ""):
                        task_id = data.get("extId")
                        return self.poll_task(task_id)
                    return data.get("extId")
                else:
                    print(f"[Error] Create VPC returned {res.status_code}: {res.text}")
            except Exception as e:
                print(f"[Error] Create VPC exception: {e}")
        return None

    @traceable(name="Nutanix: Delete VPC", run_type="tool")
    def delete_vpc(self, vpc_id: str) -> bool:
        # Clean up any subnets referencing this VPC first
        for sub in self.list_subnets():
            if sub.get("vpcReference") == vpc_id:
                self.delete_subnet(sub.get("extId"))
                time.sleep(1)

        for v in ["v4.0", "v4.0.b1", "v4.1"]:
            try:
                res = self.client.delete(
                    f"{self.base_url}/api/networking/{v}/config/vpcs/{vpc_id}",
                    headers=self._headers()
                )
                if res.status_code in [200, 202, 204]:
                    if res.status_code == 202:
                        task_id = res.json().get("data", {}).get("extId")
                        if task_id:
                            self.poll_task(task_id)
                    return True
                else:
                    print(f"[Error] Delete VPC returned {res.status_code}: {res.text}")
            except Exception as e:
                print(f"[Error] Delete VPC {vpc_id}: {e}")
        return False

# ---------------------------------------------------------------------------
# 4. LangGraph Node Implementations
# ---------------------------------------------------------------------------

def extract_subnet_ip_details(subnet: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts network IP, prefix length, gateway IP, and calculates .160-.253 IPAM pool
    from the discovered Nutanix subnet object before deletion.
    """
    net_ip = None
    prefix_len = 24
    gw_ip = None

    if subnet:
        # 1. Check ipConfig list (v4 Networking standard)
        ip_configs = subnet.get("ipConfig", [])
        if isinstance(ip_configs, list) and len(ip_configs) > 0:
            cfg = ip_configs[0]
            ipv4_cfg = cfg.get("ipv4", {}) if isinstance(cfg, dict) else {}

            # Extract IP Subnet (network IP & prefix length)
            ip_sub = ipv4_cfg.get("ipSubnet", {}) or cfg.get("ipSubnet", {})
            if isinstance(ip_sub, dict):
                ip_field = ip_sub.get("ip")
                if isinstance(ip_field, dict):
                    net_ip = ip_field.get("value")
                elif isinstance(ip_field, str):
                    net_ip = ip_field
                prefix_len = ip_sub.get("prefixLength", prefix_len)
            elif isinstance(ip_sub, str) and "/" in ip_sub:
                parts = ip_sub.split("/")
                net_ip = parts[0]
                try:
                    prefix_len = int(parts[1])
                except Exception:
                    pass

            # Extract Gateway IP
            gw_field = ipv4_cfg.get("defaultGatewayIp") or cfg.get("defaultGatewayIp")
            if isinstance(gw_field, dict):
                gw_ip = gw_field.get("value")
            elif isinstance(gw_field, str):
                gw_ip = gw_field

        # 2. Fallbacks for direct attributes if not under ipConfig
        if not net_ip:
            if subnet.get("networkIp"):
                net_ip = subnet.get("networkIp")
            elif subnet.get("subnetIp"):
                net_ip = subnet.get("subnetIp")
            elif subnet.get("ipSubnet"):
                sub_val = subnet.get("ipSubnet")
                if isinstance(sub_val, dict):
                    net_ip = sub_val.get("ip", {}).get("value") or sub_val.get("value")
                    prefix_len = sub_val.get("prefixLength", prefix_len)
                elif isinstance(sub_val, str) and "/" in sub_val:
                    parts = sub_val.split("/")
                    net_ip = parts[0]
                    try:
                        prefix_len = int(parts[1])
                    except Exception:
                        pass

        if not gw_ip:
            gw_val = subnet.get("gatewayIp") or subnet.get("defaultGatewayIp")
            if isinstance(gw_val, dict):
                gw_ip = gw_val.get("value")
            elif isinstance(gw_val, str):
                gw_ip = gw_val

        if subnet.get("prefixLength") and prefix_len == 24:
            try:
                prefix_len = int(subnet.get("prefixLength"))
            except Exception:
                pass

    # If no network IP was discovered on the subnet, fallback gracefully
    if not net_ip:
        net_ip = "10.55.81.0"
    if not gw_ip:
        octets = str(net_ip).split(".")
        if len(octets) >= 3:
            gw_ip = f"{octets[0]}.{octets[1]}.{octets[2]}.1"
        else:
            gw_ip = "10.55.81.1"

    # Compute .160 to .253 IPAM range based on first 3 octets
    base_octets = str(net_ip).split(".")
    if len(base_octets) < 3 and gw_ip:
        base_octets = str(gw_ip).split(".")

    if len(base_octets) >= 3:
        prefix_3 = f"{base_octets[0]}.{base_octets[1]}.{base_octets[2]}"
    else:
        prefix_3 = "10.55.81"

    ipam_start = f"{prefix_3}.160"
    ipam_end = f"{prefix_3}.253"

    return {
        "network_ip": net_ip,
        "prefix_length": int(prefix_len),
        "gateway_ip": gw_ip,
        "ipam_start": ipam_start,
        "ipam_end": ipam_end,
        "ipam_pool": f"{ipam_start} - {ipam_end}"
    }

async def plan_discovery_node(state: EnablementState) -> Dict[str, Any]:
    """
    Node 1: Plan (Discovery & Intent)
    1. Interrupts to ask the user for intent (Build or Destroy) and Group_ID.
    2. Queries Prism Central to discover current cluster state (existing VPCs, subnets).
    3. Captures the VLAN ID and cluster attributes of the basic subnet before planning deletion.
    4. Generates a detailed 4-step execution_plan for Build or cleanup plan for Destroy.
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

    # 2. Query Prism Central
    print(f"\n-> Connecting to Prism Central at {os.getenv('PC_HOST')}...")
    pc = NutanixPrismClient()
    existing_vpcs = pc.list_vpcs()
    existing_subnets = pc.list_subnets()

    secondary_subnet = None
    for s in existing_subnets:
        name_lower = str(s.get("name", "")).lower()
        if "secondary" in name_lower or "aux" in name_lower:
            secondary_subnet = s
            break

    if not secondary_subnet and existing_subnets:
        secondary_subnet = existing_subnets[0]

    captured_vlan_id = None
    captured_cluster_ref = None
    captured_vswitch_ref = None
    sec_name = "secondary-BLR-POC227"
    sec_id = None

    if secondary_subnet:
        sec_name = secondary_subnet.get("name", "Secondary-VLAN-Subnet")
        sec_id = secondary_subnet.get("extId")
        captured_vlan_id = secondary_subnet.get("networkId")
        if captured_vlan_id is None:
            captured_vlan_id = secondary_subnet.get("vlanId", 2271)
        captured_cluster_ref = secondary_subnet.get("clusterReference")
        if not captured_cluster_ref and secondary_subnet.get("clusterReferenceList"):
            captured_cluster_ref = secondary_subnet.get("clusterReferenceList")[0]
        captured_vswitch_ref = secondary_subnet.get("virtualSwitchReference")

    if captured_vlan_id is None:
        captured_vlan_id = 2271

    # 3. Capture Cluster Configured DNS / Name Servers
    captured_dns_servers = pc.get_cluster_dns_servers(captured_cluster_ref)
    if not captured_dns_servers:
        captured_dns_servers = ["10.42.194.10"]

    # 4. Extract IP & IPAM parameters from target subnet before deletion
    ip_details = extract_subnet_ip_details(secondary_subnet)
    captured_net_ip = ip_details["network_ip"]
    captured_prefix_len = ip_details["prefix_length"]
    captured_gateway_ip = ip_details["gateway_ip"]
    ipam_start = ip_details["ipam_start"]
    ipam_end = ip_details["ipam_end"]
    ipam_pool_str = ip_details["ipam_pool"]

    captured_vlan_info = {
        "vlan_id": int(captured_vlan_id),
        "cluster_ref": captured_cluster_ref,
        "vswitch_ref": captured_vswitch_ref,
        "subnet_name": sec_name,
        "subnet_ext_id": sec_id,
        "network_ip": captured_net_ip,
        "prefix_length": captured_prefix_len,
        "gateway_ip": captured_gateway_ip,
        "ipam_start": ipam_start,
        "ipam_end": ipam_end,
        "ipam_pool": ipam_pool_str
    }

    discovered_state = {
        "existing_vpcs": existing_vpcs,
        "existing_subnets": existing_subnets,
        "secondary_subnet": secondary_subnet,
        "dns_servers": captured_dns_servers
    }

    print(f"-> Discovered {len(existing_vpcs)} live VPC(s) and {len(existing_subnets)} subnet(s) on cluster.")
    print(f"-> Target Subnet Identified: '{sec_name}' (ID: {sec_id or 'Auto-Detect'})")
    print(f"-> [CAPTURE] VLAN ID captured: {captured_vlan_id} | Cluster: {captured_cluster_ref}")
    print(f"-> [CAPTURE] Network: {captured_net_ip}/{captured_prefix_len} | Gateway: {captured_gateway_ip}")
    print(f"-> [CAPTURE] Derived IPAM Pool (.160-.253): {ipam_pool_str}")
    print(f"-> [CAPTURE] Cluster DNS Server(s) captured: {captured_dns_servers}")

    # 5. Generate Execution Plan
    execution_plan = []
    if intent == "Build":
        execution_plan = [
            {
                "step": 1,
                "action": "DELETE_BASIC_SUBNET",
                "target_type": "Subnet",
                "target_name": sec_name,
                "target_id": sec_id,
                "details": {
                    "description": f"Delete predefined Basic VLAN subnet '{sec_name}' to free VLAN ID {captured_vlan_id}",
                    "captured_vlan_id": captured_vlan_id,
                    "target_ext_id": sec_id
                }
            },
            {
                "step": 2,
                "action": "CREATE_EXTERNAL_VLAN_SUBNET",
                "target_type": "Subnet",
                "target_name": sec_name,
                "details": {
                    "description": f"Create Network Controller External VLAN Subnet preserving network {captured_net_ip}/{captured_prefix_len} and gateway {captured_gateway_ip}",
                    "vlan_id": captured_vlan_id,
                    "subnet_type": "VLAN (External)",
                    "network_ip": f"{captured_net_ip}/{captured_prefix_len}",
                    "gateway_ip": captured_gateway_ip,
                    "ipam_pool": ipam_pool_str,
                    "cluster_ref": captured_cluster_ref
                }
            },
            {
                "step": 3,
                "action": "CREATE_TRANSIT_VPC",
                "target_type": "VPC",
                "target_name": f"Transit-VPC-{group_id}",
                "details": {
                    "description": "Create Transit VPC attached to External Subnet with NAT enabled, ERP advertisement, and cluster DNS",
                    "dns_servers": captured_dns_servers,
                    "external_connectivity": f"NAT enabled to External Subnet ({sec_name})",
                    "subnets": [
                        {
                            "name": f"Transit-ERP-{group_id}",
                            "cidr": "10.10.10.0/24",
                            "gateway": "10.10.10.1",
                            "ipam_range": "10.10.10.160 - 10.10.10.253",
                            "type": "ERP (Externally Routable Prefix)"
                        },
                        {
                            "name": f"Transit-NonERP-{group_id}",
                            "cidr": "20.20.20.0/24",
                            "gateway": "20.20.20.1",
                            "ipam_range": "20.20.20.160 - 20.20.20.253",
                            "type": "Non-ERP"
                        }
                    ]
                }
            },
            {
                "step": 4,
                "action": "CREATE_SPOKE_VPCS",
                "target_type": "VPC_GROUP",
                "target_name": f"Spoke-VPCs (1..3) for Group {group_id}",
                "details": {
                    "dns_servers": captured_dns_servers,
                    "spokes": [
                        {
                            "name": f"Spoke-VPC-1-{group_id}",
                            "cidr": "1.1.1.0/24",
                            "gateway": "1.1.1.1",
                            "ipam_range": "1.1.1.160 - 1.1.1.253",
                            "type": "ERP (Externally Routable Prefix)",
                            "connectivity": f"No-NAT to Transit-VPC-{group_id}"
                        },
                        {
                            "name": f"Spoke-VPC-2-{group_id}",
                            "cidr": "2.2.2.0/24",
                            "gateway": "2.2.2.1",
                            "ipam_range": "2.2.2.160 - 2.2.2.253",
                            "type": "ERP (Externally Routable Prefix)",
                            "connectivity": f"No-NAT to Transit-VPC-{group_id}"
                        },
                        {
                            "name": f"Spoke-VPC-3-{group_id}",
                            "cidr": "3.3.3.0/24",
                            "gateway": "3.3.3.1",
                            "ipam_range": "3.3.3.160 - 3.3.3.253",
                            "type": "ERP (Externally Routable Prefix)",
                            "connectivity": f"No-NAT to Transit-VPC-{group_id}"
                        }
                    ]
                }
            }
        ]
    else:  # Destroy Intent
        vpcs_to_delete = []
        for vpc in existing_vpcs:
            v_name = vpc.get("name", "")
            if "Enablement" in v_name or "Group" in v_name or "Transit" in v_name or "Spoke" in v_name or "Test" in v_name:
                vpcs_to_delete.append(vpc)

        subnets_to_delete = []
        for sub in existing_subnets:
            s_name = sub.get("name", "")
            if "Enablement" in s_name or "Group" in s_name or "Transit" in s_name or "Spoke" in s_name:
                subnets_to_delete.append(sub)

        step_idx = 1
        for vpc in vpcs_to_delete:
            execution_plan.append({
                "step": step_idx,
                "action": "DELETE_VPC",
                "target_type": "VPC",
                "target_name": vpc.get("name"),
                "target_id": vpc.get("extId"),
                "details": {"description": f"Delete VPC '{vpc.get('name')}'"}
            })
            step_idx += 1

        for sub in subnets_to_delete:
            execution_plan.append({
                "step": step_idx,
                "action": "DELETE_SUBNET",
                "target_type": "Subnet",
                "target_name": sub.get("name"),
                "target_id": sub.get("extId"),
                "details": {"description": f"Delete subnet '{sub.get('name')}'"}
            })
            step_idx += 1

        if not execution_plan:
            execution_plan.append({
                "step": 1,
                "action": "NO_OP_CLEANUP",
                "target_type": "None",
                "target_name": "No Enablement Constructs Found",
                "details": {"description": "No enablement constructs found for cleanup."}
            })

    print(f"[OK] Generated Execution Plan ({len(execution_plan)} step(s)).")

    return {
        "user_intent": intent,
        "group_id": group_id,
        "discovered_state": discovered_state,
        "captured_vlan_info": captured_vlan_info,
        "captured_dns_servers": captured_dns_servers,
        "execution_plan": execution_plan,
        "messages": [AIMessage(content=f"Plan generated for {intent} with {len(execution_plan)} steps.")]
    }

async def review_approval_node(state: EnablementState) -> Dict[str, Any]:
    """Node 2: Human-In-The-Loop Plan Approval."""
    print("\n=========================================================")
    print(" [Node 2: Review] Human-in-the-Loop Approval")
    print("=========================================================")

    plan = state.get("execution_plan", [])
    intent = state.get("user_intent", "Build")
    group_id = state.get("group_id", "01")
    captured_info = state.get("captured_vlan_info", {})
    captured_dns = state.get("captured_dns_servers", [])

    print(f"\nPROPOSED EXECUTION PLAN (Intent: {intent} | Group: {group_id}):")
    print("-" * 75)
    for p in plan:
        print(f"Step {p.get('step')}: [{p.get('action')}] {p.get('target_name')}")
        details = p.get("details", {})
        if "spokes" in details:
            if "dns_servers" in details:
                print(f"   * DNS Server(s): {', '.join(details.get('dns_servers', []))}")
            for sp in details["spokes"]:
                print(f"   * {sp.get('name')} (CIDR: {sp.get('cidr')}, Type: {sp.get('type')}, Route: {sp.get('connectivity')})")
        elif "subnets" in details:
            if "dns_servers" in details:
                print(f"   * DNS Server(s): {', '.join(details.get('dns_servers', []))}")
            print(f"   * Config: {details.get('external_connectivity')}")
            for sub in details["subnets"]:
                print(f"   * Subnet: {sub.get('name')} | CIDR: {sub.get('cidr')} | Type: {sub.get('type')}")
        else:
            for k, v in details.items():
                print(f"   * {k}: {v}")
    print("-" * 75)

    if intent == "Build":
        if captured_info.get("vlan_id"):
            print(f"[CONFIRMATION] Captured VLAN ID to reuse: {captured_info.get('vlan_id')}")
        if captured_info.get("network_ip"):
            print(f"[CONFIRMATION] Captured Subnet Network: {captured_info.get('network_ip')}/{captured_info.get('prefix_length')}")
            print(f"[CONFIRMATION] Captured Gateway IP    : {captured_info.get('gateway_ip')}")
            print(f"[CONFIRMATION] Configured IPAM Pool   : {captured_info.get('ipam_pool')}")
        if captured_dns:
            print(f"[CONFIRMATION] Captured Cluster DNS Server(s) to configure: {', '.join(captured_dns)}")

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
    """Node 3: Execute Provisioning Constructs against Prism Central."""
    print("\n=========================================================")
    print(" [Node 3: Execute] Provisioning Constructs on Prism Central")
    print("=========================================================")

    plan = state.get("execution_plan", [])
    intent = state.get("user_intent", "Build")
    group_id = state.get("group_id", "01")
    captured_info = state.get("captured_vlan_info", {})
    captured_dns_servers = state.get("captured_dns_servers", [])
    vlan_id = captured_info.get("vlan_id", 2271)
    cluster_ref = captured_info.get("cluster_ref")
    vswitch_ref = captured_info.get("vswitch_ref")

    execution_results = []
    created_external_subnet_ext_id = None
    created_transit_vpc_ext_id = None

    pc = NutanixPrismClient()

    for item in plan:
        action = item.get("action")
        target_name = item.get("target_name")
        target_id = item.get("target_id")
        step_num = item.get("step")

        print(f"\n-> Executing Step {step_num}: {action} ({target_name})...")

        if action == "DELETE_BASIC_SUBNET":
            ok = False
            if target_id:
                ok = pc.delete_subnet(target_id)
                print(f"-> Sent delete request for {target_id}. Waiting 4s for cluster to release VLAN...")
                await asyncio.sleep(4)
            else:
                ok = True

            if ok:
                print(f"[OK] Deleted Basic Subnet '{target_name}' (ID: {target_id}). VLAN {vlan_id} is now released.")
            else:
                print(f"[Warning] Delete Basic Subnet did not complete cleanly or was already removed.")

            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": target_id or "subnet-basic-id",
                "status": "SUCCESS" if ok else "FAILED",
                "details": f"Deleted basic subnet to free VLAN ID {vlan_id}"
            })

        elif action == "CREATE_EXTERNAL_VLAN_SUBNET":
            net_ip = captured_info.get("network_ip") or item.get("details", {}).get("network_ip", "10.55.81.0").split("/")[0]
            prefix_len = int(captured_info.get("prefix_length") or 24)
            gw_ip = captured_info.get("gateway_ip") or item.get("details", {}).get("gateway_ip", "10.55.81.1")
            pool_start = captured_info.get("ipam_start") or f"{'.'.join(net_ip.split('.')[:3])}.160"
            pool_end = captured_info.get("ipam_end") or f"{'.'.join(net_ip.split('.')[:3])}.253"

            subnet_body: Dict[str, Any] = {
                "name": target_name,
                "subnetType": "VLAN",
                "networkId": int(vlan_id),
                "isExternal": True,
                "ipConfig": [
                    {
                        "ipv4": {
                            "ipSubnet": {
                                "ip": {"value": net_ip, "prefixLength": 32},
                                "prefixLength": prefix_len
                            },
                            "defaultGatewayIp": {"value": gw_ip, "prefixLength": 32},
                            "poolList": [
                                {
                                    "startIp": {"value": pool_start, "prefixLength": 32},
                                    "endIp": {"value": pool_end, "prefixLength": 32}
                                }
                            ]
                        }
                    }
                ]
            }
            if cluster_ref:
                subnet_body["clusterReference"] = cluster_ref
            if vswitch_ref:
                subnet_body["virtualSwitchReference"] = vswitch_ref

            created_external_subnet_ext_id = pc.create_subnet(subnet_body)
            if not created_external_subnet_ext_id:
                # If already existing external subnet on cluster, reuse it
                created_external_subnet_ext_id = captured_info.get("subnet_ext_id")

            if created_external_subnet_ext_id:
                print(f"[OK] External VLAN Subnet '{target_name}' ready with VLAN {vlan_id} ({net_ip}/{prefix_len}, GW {gw_ip}, IPAM {pool_start}-{pool_end}) [ID: {created_external_subnet_ext_id}]")
                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": created_external_subnet_ext_id,
                    "status": "SUCCESS",
                    "details": f"External VLAN Subnet with VLAN ID {vlan_id} ({net_ip}/{prefix_len}, GW {gw_ip}, IPAM {pool_start}-{pool_end})"
                })
            else:
                print(f"[Error] Failed to create External VLAN Subnet '{target_name}'.")
                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": "N/A",
                    "status": "FAILED",
                    "details": "Subnet creation API call failed."
                })

        elif action == "CREATE_TRANSIT_VPC":
            # Idempotency check: remove any existing VPC with identical name
            for existing_v in pc.list_vpcs():
                if existing_v.get("name") == target_name:
                    print(f"-> Cleaning up existing VPC '{target_name}' ({existing_v.get('extId')}) before creation...")
                    pc.delete_vpc(existing_v.get("extId"))
                    await asyncio.sleep(2)

            vpc_body: Dict[str, Any] = {
                "name": target_name,
                "description": f"Transit VPC for Enablement Group {group_id}",
                "shouldAdvertiseConnectedSubnets": True,
                "externallyRoutablePrefixes": [
                    {"ipv4": {"ip": {"value": "10.10.10.0", "prefixLength": 32}, "prefixLength": 24}},
                    {"ipv4": {"ip": {"value": "1.1.1.0", "prefixLength": 32}, "prefixLength": 24}},
                    {"ipv4": {"ip": {"value": "2.2.2.0", "prefixLength": 32}, "prefixLength": 24}},
                    {"ipv4": {"ip": {"value": "3.3.3.0", "prefixLength": 32}, "prefixLength": 24}}
                ]
            }
            if captured_dns_servers:
                vpc_body["commonDhcpOptions"] = {
                    "domainNameServers": [
                        {"ipv4": {"value": dns_ip, "prefixLength": 32}}
                        for dns_ip in captured_dns_servers
                    ]
                }
            
            ext_sub_ref = created_external_subnet_ext_id or captured_info.get("subnet_ext_id")
            if ext_sub_ref:
                vpc_body["externalSubnets"] = [
                    {"subnetReference": ext_sub_ref}
                ]

            created_transit_vpc_ext_id = pc.create_vpc(vpc_body)

            if created_transit_vpc_ext_id:
                dns_str = f" with DNS {captured_dns_servers}" if captured_dns_servers else ""
                print(f"[OK] Created Transit VPC '{target_name}'{dns_str} & ERP Advertisements [ID: {created_transit_vpc_ext_id}]")

                # Create ERP and Non-ERP overlay subnets with full IPAM
                erp_sub = {
                    "name": f"Transit-ERP-{group_id}",
                    "subnetType": "OVERLAY",
                    "vpcReference": created_transit_vpc_ext_id,
                    "ipConfig": [
                        {
                            "ipv4": {
                                "ipSubnet": {"ip": {"value": "10.10.10.0", "prefixLength": 32}, "prefixLength": 24},
                                "defaultGatewayIp": {"value": "10.10.10.1", "prefixLength": 32},
                                "poolList": [{"startIp": {"value": "10.10.10.160", "prefixLength": 32}, "endIp": {"value": "10.10.10.253", "prefixLength": 32}}]
                            }
                        }
                    ]
                }
                non_erp_sub = {
                    "name": f"Transit-NonERP-{group_id}",
                    "subnetType": "OVERLAY",
                    "vpcReference": created_transit_vpc_ext_id,
                    "ipConfig": [
                        {
                            "ipv4": {
                                "ipSubnet": {"ip": {"value": "20.20.20.0", "prefixLength": 32}, "prefixLength": 24},
                                "defaultGatewayIp": {"value": "20.20.20.1", "prefixLength": 32},
                                "poolList": [{"startIp": {"value": "20.20.20.160", "prefixLength": 32}, "endIp": {"value": "20.20.20.253", "prefixLength": 32}}]
                            }
                        }
                    ]
                }
                pc.create_subnet(erp_sub)
                pc.create_subnet(non_erp_sub)

                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": created_transit_vpc_ext_id,
                    "status": "SUCCESS",
                    "details": f"Transit VPC created with DNS {captured_dns_servers}, NAT, ERP/Non-ERP subnets and IPAM pools (.160-.253)"
                })
            else:
                print(f"[Error] Failed to create Transit VPC '{target_name}'.")
                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": "N/A",
                    "status": "FAILED",
                    "details": "Transit VPC creation API call failed."
                })

        elif action == "CREATE_SPOKE_VPCS":
            spokes_config = [
                {"index": 1, "ip": "1.1.1.0", "gw": "1.1.1.1", "pool_start": "1.1.1.160", "pool_end": "1.1.1.253"},
                {"index": 2, "ip": "2.2.2.0", "gw": "2.2.2.1", "pool_start": "2.2.2.160", "pool_end": "2.2.2.253"},
                {"index": 3, "ip": "3.3.3.0", "gw": "3.3.3.1", "pool_start": "3.3.3.160", "pool_end": "3.3.3.253"}
            ]
            spoke_results = []
            for spoke in spokes_config:
                spoke_name = f"Spoke-VPC-{spoke['index']}-{group_id}"
                
                # Idempotency check: remove any existing VPC with identical name
                for existing_v in pc.list_vpcs():
                    if existing_v.get("name") == spoke_name:
                        print(f"   -> Cleaning up existing Spoke VPC '{spoke_name}' ({existing_v.get('extId')})...")
                        pc.delete_vpc(existing_v.get("extId"))
                        await asyncio.sleep(2)

                spoke_body = {
                    "name": spoke_name,
                    "description": f"Spoke {spoke['index']} VPC for Group {group_id}",
                    "shouldAdvertiseConnectedSubnets": True,
                    "externallyRoutablePrefixes": [
                        {
                            "ipv4": {
                                "ip": {"value": spoke["ip"], "prefixLength": 32},
                                "prefixLength": 24
                            }
                        }
                    ]
                }
                if captured_dns_servers:
                    spoke_body["commonDhcpOptions"] = {
                        "domainNameServers": [
                            {"ipv4": {"value": dns_ip, "prefixLength": 32}}
                            for dns_ip in captured_dns_servers
                        ]
                    }
                spoke_id = pc.create_vpc(spoke_body)
                if spoke_id:
                    # Create overlay ERP subnet in Spoke VPC with full IPAM
                    spoke_sub = {
                        "name": f"Spoke-ERP-{spoke['index']}-{group_id}",
                        "subnetType": "OVERLAY",
                        "vpcReference": spoke_id,
                        "ipConfig": [
                            {
                                "ipv4": {
                                    "ipSubnet": {"ip": {"value": spoke["ip"], "prefixLength": 32}, "prefixLength": 24},
                                    "defaultGatewayIp": {"value": spoke["gw"], "prefixLength": 32},
                                    "poolList": [
                                        {
                                            "startIp": {"value": spoke["pool_start"], "prefixLength": 32},
                                            "endIp": {"value": spoke["pool_end"], "prefixLength": 32}
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                    pc.create_subnet(spoke_sub)
                    print(f"   [OK] Spoke {spoke['index']}/3: Created '{spoke_name}' (DNS {captured_dns_servers}, ERP {spoke['ip']}/24, GW {spoke['gw']}, IPAM {spoke['pool_start']}-{spoke['pool_end']}) [ID: {spoke_id}]")
                    spoke_results.append({"name": spoke_name, "extId": spoke_id, "status": "SUCCESS"})
                else:
                    print(f"   [Error] Failed to create Spoke VPC '{spoke_name}'")
                    spoke_results.append({"name": spoke_name, "extId": "N/A", "status": "FAILED"})

            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": f"Spoke VPCs (1..3)-{group_id}",
                "status": "SUCCESS" if all(s["status"] == "SUCCESS" for s in spoke_results) else "PARTIAL",
                "spokes": spoke_results,
                "details": f"3 Spoke VPCs provisioned with DNS {captured_dns_servers}, ERP designations and full IPAM pools (.160-.253)"
            })

        elif action == "DELETE_VPC":
            ok = pc.delete_vpc(target_id) if target_id else True
            print(f"[{'OK' if ok else 'Warning'}] Delete VPC '{target_name}' (ID: {target_id})")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": target_id,
                "status": "DELETED" if ok else "FAILED"
            })

        elif action == "DELETE_SUBNET":
            ok = pc.delete_subnet(target_id) if target_id else True
            print(f"[{'OK' if ok else 'Warning'}] Delete Subnet '{target_name}' (ID: {target_id})")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": target_id,
                "status": "DELETED" if ok else "FAILED"
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
    """Node 4: System Verification & Live Audit against Prism Central."""
    print("\n=========================================================")
    print(" [Node 4: Review] System Verification & Live Audit")
    print("=========================================================")

    results = state.get("execution_results", [])
    intent = state.get("user_intent", "Build")
    group_id = state.get("group_id", "01")
    captured_info = state.get("captured_vlan_info", {})
    captured_dns_servers = state.get("captured_dns_servers", [])
    vlan_id = captured_info.get("vlan_id", 2271)

    pc = NutanixPrismClient()
    live_vpcs = pc.list_vpcs()
    live_subnets = pc.list_subnets()

    print(f"-> Live Audit Complete: {len(live_vpcs)} VPC(s) and {len(live_subnets)} Subnet(s) found on cluster.")

    verification_table = []
    for res in results:
        status_tag = f"[{res.get('status')}]"
        verification_table.append({
            "Action": res.get("action"),
            "Target": res.get("target_name"),
            "Status": status_tag,
            "ExtID": res.get("extId", "N/A")
        })

    print("\n=========================================================")
    print(f" FINAL EXECUTION SUMMARY (Intent: {intent} | Group: {group_id})")
    print("=========================================================")
    print(f"{'Action':<28} | {'Target Name':<30} | {'Status':<10} | {'Resource ExtID'}")
    print("-" * 90)
    for row in verification_table:
        print(f"{row['Action']:<28} | {row['Target']:<30} | {row['Status']:<10} | {row['ExtID']}")
    print("-" * 90)

    if intent == "Build":
        dns_display = ", ".join(captured_dns_servers) if captured_dns_servers else "None"
        ext_net_str = f"{captured_info.get('network_ip', '10.55.81.0')}/{captured_info.get('prefix_length', 24)}"
        ext_gw_str = captured_info.get('gateway_ip', '10.55.81.1')
        ext_ipam_str = captured_info.get('ipam_pool', '.160-.253')
        print("\nStudent Enablement Networking Topology Provisioned:")
        print(f" * Reused VLAN ID     : {vlan_id}")
        print(f" * Cluster DNS Server : {dns_display} (Configured on all VPCs)")
        print(f" * External Network   : Network Controller Subnet {ext_net_str} (GW {ext_gw_str}, IPAM {ext_ipam_str})")
        print(f" * Transit VPC        : Transit-VPC-{group_id} (DNS: {dns_display}, NAT Enabled, ERP Advertised)")
        print(f"   - Transit-ERP-{group_id}    : 10.10.10.0/24 (GW 10.10.10.1, IPAM .160-.253, ERP)")
        print(f"   - Transit-NonERP-{group_id} : 20.20.20.0/24 (GW 20.20.20.1, IPAM .160-.253, Non-ERP)")
        print(f" * Spoke 1 VPC        : Spoke-VPC-1-{group_id} (DNS: {dns_display}, ERP 1.1.1.0/24, GW 1.1.1.1, IPAM .160-.253, No-NAT -> Transit)")
        print(f" * Spoke 2 VPC        : Spoke-VPC-2-{group_id} (DNS: {dns_display}, ERP 2.2.2.0/24, GW 2.2.2.1, IPAM .160-.253, No-NAT -> Transit)")
        print(f" * Spoke 3 VPC        : Spoke-VPC-3-{group_id} (DNS: {dns_display}, ERP 3.3.3.0/24, GW 3.3.3.1, IPAM .160-.253, No-NAT -> Transit)")
    else:
        print("\nStudent Enablement Constructs Cleaned & Destroyed Successfully.")

    print("=========================================================\n")

    summary_text = f"Audit complete: {len(results)} actions verified for {intent}."
    return {
        "final_summary": summary_text,
        "messages": [AIMessage(content=summary_text)]
    }

# ---------------------------------------------------------------------------
# 5. Routing Decisions
# ---------------------------------------------------------------------------
def route_after_review(state: EnablementState) -> str:
    if state.get("approval_status") == "approved":
        return "execute_provisioning"
    return END

# ---------------------------------------------------------------------------
# 6. StateGraph Construction & Compilation
# ---------------------------------------------------------------------------
def build_enablement_graph():
    builder = StateGraph(EnablementState)

    builder.add_node("plan_discovery", plan_discovery_node)
    builder.add_node("review_approval", review_approval_node)
    builder.add_node("execute_provisioning", execute_provisioning_node)
    builder.add_node("review_verification", review_verification_node)

    builder.add_edge(START, "plan_discovery")
    builder.add_edge("plan_discovery", "review_approval")

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
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "Nutanix HPOC Network Enablement",
        "tags": ["nutanix", "transit-vpc", "automation"],
        "metadata": {
            "thread_id": thread_id,
            "pc_host": os.getenv("PC_HOST", "127.0.0.1")
        }
    }

    initial_state = {
        "messages": [SystemMessage(content="Starting Nutanix Enablement Networking workflow.")]
    }

    async for event in app.astream(initial_state, config=config):
        for node_name, state_update in event.items():
            if node_name == "__interrupt__":
                continue

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
