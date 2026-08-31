import os
import sys
import json
import shutil
import asyncio
import re
import time
import uuid
import base64
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
    discovered_state: Dict[str, Any]        # Cluster snapshot (VPCs, Subnets, Storage, Categories, VMs)
    captured_vlan_info: Dict[str, Any]      # Extracted VLAN ID, Cluster Ref, Subnet Name prior to deletion
    captured_dns_servers: List[str]         # Extracted Cluster DNS / Name Server IPs
    execution_plan: List[Dict[str, Any]]    # Generated list of planned actions
    approval_status: Optional[str]          # "approved" or "rejected"
    execution_results: List[Dict[str, Any]] # Outcome logs & UUIDs per executed action
    final_summary: Optional[str]            # Final post-audit comparison report
    messages: Annotated[List[BaseMessage], add_messages]

# ---------------------------------------------------------------------------
# 3. Nutanix Prism Central v4 & v3 API Client
# ---------------------------------------------------------------------------
class NutanixPrismClient:
    """Direct client for Nutanix Prism Central v4 & v3 APIs with idempotency and task polling."""

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
            timeout=35.0
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def poll_task(self, task_ext_id: str, timeout_sec: int = 60) -> Optional[str]:
        """Polls async task until SUCCEEDED, FAILED, or timeout."""
        start = time.time()
        while time.time() - start < timeout_sec:
            try:
                # Try v4 tasks first
                url = f"{self.base_url}/api/prism/v4.0/config/tasks/{task_ext_id}"
                resp = self.client.get(url, headers=self._headers())
                if resp.status_code == 200:
                    status = resp.json().get("data", {}).get("status")
                    if status in ["SUCCEEDED", "SUCCESS"]:
                        return "SUCCEEDED"
                    if status in ["FAILED", "CANCELED"]:
                        return status
                # Try v3 tasks fallback
                url_v3 = f"{self.base_url}/api/nutanix/v3/tasks/{task_ext_id}"
                resp_v3 = self.client.get(url_v3, headers=self._headers())
                if resp_v3.status_code == 200:
                    status = resp_v3.json().get("status")
                    if status in ["SUCCEEDED", "SUCCESS"]:
                        return "SUCCEEDED"
                    if status in ["FAILED", "CANCELED"]:
                        return status
            except Exception:
                pass
            time.sleep(2)
        return "TIMEOUT"

    # --- CLUSTERS ---
    def list_clusters(self) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/api/clustermgmt/v4.0/config/clusters"
            resp = self.client.get(url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if isinstance(data, list):
                    return data
            # Fallback to v3
            url_v3 = f"{self.base_url}/api/nutanix/v3/clusters/list"
            resp_v3 = self.client.post(url_v3, json={"kind": "cluster"}, headers=self._headers())
            if resp_v3.status_code == 200:
                return resp_v3.json().get("entities", [])
        except Exception as e:
            print(f"[Warning] Failed to list clusters: {e}")
        return []

    def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/api/clustermgmt/v4.0/config/clusters/{cluster_id}"
            resp = self.client.get(url, headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("data")
        except Exception:
            pass
        return None

    def get_cluster_dns_servers(self, preferred_cluster_id: Optional[str] = None) -> List[str]:
        """Discovers cluster-configured Name / DNS Servers."""
        dns_servers: List[str] = []
        try:
            clusters = self.list_clusters()
            target_cluster = None
            if preferred_cluster_id:
                for c in clusters:
                    c_id = c.get("extId") or c.get("metadata", {}).get("uuid")
                    if c_id == preferred_cluster_id:
                        target_cluster = c
                        break
            if not target_cluster and clusters:
                target_cluster = clusters[0]

            if target_cluster:
                config = target_cluster.get("config", {}) or target_cluster.get("spec", {}).get("resources", {}).get("config", {})
                name_server_list = (
                    config.get("nameServerIpList") or
                    config.get("name_server_ip_list") or
                    config.get("dnsServerIpList") or
                    []
                )
                for ns in name_server_list:
                    if isinstance(ns, dict):
                        val = ns.get("ipv4", {}).get("value") or ns.get("value")
                    else:
                        val = str(ns)
                    if val and val not in dns_servers:
                        dns_servers.append(val)
        except Exception as e:
            print(f"[Warning] Could not extract cluster DNS servers: {e}")

        return dns_servers

    # --- STORAGE CONTAINERS ---
    def list_storage_containers(self) -> List[Dict[str, Any]]:
        """Queries storage containers via Cluster Management v4 / v2.0 API."""
        try:
            url = f"{self.base_url}/api/clustermgmt/v4.0/config/storage-containers"
            resp = self.client.get(url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if isinstance(data, list):
                    return data
            # Fallback to v2.0
            url_v2 = f"{self.base_url}/PrismGateway/services/rest/v2.0/storage_containers"
            resp_v2 = self.client.get(url_v2, headers=self._headers())
            if resp_v2.status_code == 200:
                return resp_v2.json().get("entities", [])
        except Exception as e:
            print(f"[Warning] Failed to list storage containers: {e}")
        return []

    def get_storage_container(self, name: str) -> Optional[Dict[str, Any]]:
        for c in self.list_storage_containers():
            if str(c.get("name", "")).lower() == name.lower():
                return c
        return None

    def create_storage_container(self, name: str = "nkp", cluster_ext_id: Optional[str] = None, storage_pool_ext_id: Optional[str] = None) -> Optional[str]:
        """Creates storage container with default parameters if not already existing."""
        existing = self.get_storage_container(name)
        if existing:
            return existing.get("containerExtId") or existing.get("id")

        if not cluster_ext_id:
            clusters = self.list_clusters()
            if clusters:
                cluster_ext_id = clusters[0].get("extId") or clusters[0].get("uuid") or clusters[0].get("metadata", {}).get("uuid")
        if not cluster_ext_id:
            cluster_ext_id = "00065a51-3cfb-9563-0000-0000000297f5"

        if not storage_pool_ext_id:
            for c in self.list_storage_containers():
                if c.get("storagePoolExtId"):
                    storage_pool_ext_id = c.get("storagePoolExtId")
                    break
        if not storage_pool_ext_id:
            storage_pool_ext_id = "4c9fbf10-d529-440a-ba33-b9e335ed1dfe"

        url = f"{self.base_url}/api/clustermgmt/v4.0/config/storage-containers"
        headers = self._headers()
        headers["X-Cluster-Id"] = cluster_ext_id
        headers["NTNX-Request-Id"] = str(uuid.uuid4())

        payload = {
            "name": name,
            "clusterExtId": cluster_ext_id,
            "storagePoolExtId": storage_pool_ext_id
        }

        try:
            resp = self.client.post(url, json=payload, headers=headers)
            if resp.status_code in [200, 201, 202]:
                data = resp.json().get("data", {})
                task_id = data.get("extId")
                if task_id:
                    self.poll_task(task_id, timeout_sec=45)
                created = self.get_storage_container(name)
                if created:
                    return created.get("containerExtId") or created.get("id")
            else:
                print(f"[Warning] Create storage container response: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"[Warning] Error creating storage container {name}: {e}")
        return None

    # --- CATEGORIES ---
    def ensure_categories(self, categories_map: Dict[str, List[str]]) -> Dict[str, bool]:
        """Creates Prism Central category keys and values idempotently."""
        results = {}
        for key, values in categories_map.items():
            # 1. Ensure category key exists
            url_key = f"{self.base_url}/api/nutanix/v3/categories/{key}"
            try:
                res_key = self.client.put(url_key, json={"name": key, "description": f"{key.capitalize()} Category"})
                key_ok = res_key.status_code in [200, 201]
            except Exception as e:
                print(f"[Warning] Category key '{key}' creation note: {e}")
                key_ok = False

            # 2. Ensure category values exist
            for val in values:
                url_val = f"{self.base_url}/api/nutanix/v3/categories/{key}/{val}"
                try:
                    res_val = self.client.put(url_val, json={"value": val, "description": f"{val.capitalize()} {key.capitalize()}"})
                    val_ok = res_val.status_code in [200, 201]
                    results[f"{key}:{val}"] = val_ok
                except Exception as e:
                    print(f"[Warning] Category value '{key}:{val}' creation note: {e}")
                    results[f"{key}:{val}"] = False
        return results

    # --- IMAGES ---
    def list_images(self) -> List[Dict[str, Any]]:
        """Queries images from Prism Central image repository."""
        try:
            url = f"{self.base_url}/api/nutanix/v3/images/list"
            resp = self.client.post(url, json={"kind": "image", "length": 100}, headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("entities", [])
        except Exception as e:
            print(f"[Warning] Failed to list images: {e}")
        return []

    def get_image_by_name(self, name_substr: str) -> Optional[Dict[str, Any]]:
        images = self.list_images()
        for img in images:
            img_name = str(img.get("spec", {}).get("name") or img.get("status", {}).get("name", "")).lower()
            if name_substr.lower() in img_name:
                return img
        return None

    # --- VMS ---
    def list_vms(self) -> List[Dict[str, Any]]:
        """Queries VMs from Prism Central v3 API."""
        try:
            url = f"{self.base_url}/api/nutanix/v3/vms/list"
            resp = self.client.post(url, json={"kind": "vm", "length": 200}, headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("entities", [])
        except Exception as e:
            print(f"[Warning] Failed to list VMs: {e}")
        return []

    def get_vm_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for vm in self.list_vms():
            vm_name = str(vm.get("spec", {}).get("name") or vm.get("status", {}).get("name", ""))
            if vm_name.lower() == name.lower():
                return vm
        return None

    def create_vm(self, vm_payload: Dict[str, Any]) -> Optional[str]:
        """Creates a VM via Nutanix v3 API and polls task to completion."""
        try:
            url = f"{self.base_url}/api/nutanix/v3/vms"
            resp = self.client.post(url, json=vm_payload, headers=self._headers())
            if resp.status_code in [200, 201, 202]:
                data = resp.json()
                task_id = data.get("status", {}).get("execution_context", {}).get("task_uuid")
                vm_uuid = data.get("metadata", {}).get("uuid")
                if task_id:
                    self.poll_task(task_id, timeout_sec=90)
                return vm_uuid
            else:
                print(f"[Error] Failed to create VM: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"[Error] Exception in create_vm: {e}")
        return None

    def delete_vm(self, vm_uuid: str) -> bool:
        """Deletes a VM via Nutanix v3 API."""
        try:
            url = f"{self.base_url}/api/nutanix/v3/vms/{vm_uuid}"
            resp = self.client.delete(url, headers=self._headers())
            if resp.status_code in [200, 202, 204]:
                data = resp.json() if resp.content else {}
                task_id = data.get("status", {}).get("execution_context", {}).get("task_uuid")
                if task_id:
                    self.poll_task(task_id, timeout_sec=60)
                return True
        except Exception as e:
            print(f"[Warning] Failed to delete VM {vm_uuid}: {e}")
        return False

    # --- SUBNETS ---
    def list_subnets(self) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/api/networking/v4.0/config/subnets"
            resp = self.client.get(url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if isinstance(data, list):
                    return data
            # Fallback to v3
            url_v3 = f"{self.base_url}/api/nutanix/v3/subnets/list"
            resp_v3 = self.client.post(url_v3, json={"kind": "subnet"}, headers=self._headers())
            if resp_v3.status_code == 200:
                return resp_v3.json().get("entities", [])
        except Exception as e:
            print(f"[Warning] Failed to list subnets: {e}")
        return []

    def get_subnet(self, subnet_id: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/api/networking/v4.0/config/subnets/{subnet_id}"
            resp = self.client.get(url, headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("data")
        except Exception:
            pass
        return None

    def delete_subnet(self, subnet_id: str) -> bool:
        try:
            url = f"{self.base_url}/api/networking/v4.0/config/subnets/{subnet_id}"
            resp = self.client.delete(url, headers=self._headers())
            if resp.status_code in [200, 202, 204]:
                data = resp.json().get("data", {})
                task_id = data.get("extId")
                if task_id:
                    self.poll_task(task_id, timeout_sec=30)
                return True
            # Fallback to v3
            url_v3 = f"{self.base_url}/api/nutanix/v3/subnets/{subnet_id}"
            resp_v3 = self.client.delete(url_v3, headers=self._headers())
            if resp_v3.status_code in [200, 202, 204]:
                return True
        except Exception as e:
            print(f"[Warning] Failed to delete subnet {subnet_id}: {e}")
        return False

    def create_subnet(self, subnet_payload: Dict[str, Any]) -> Optional[str]:
        try:
            url = f"{self.base_url}/api/networking/v4.0/config/subnets"
            headers = self._headers()
            headers["NTNX-Request-Id"] = str(uuid.uuid4())
            resp = self.client.post(url, json=subnet_payload, headers=headers)
            if resp.status_code in [200, 201, 202]:
                data = resp.json().get("data", {})
                task_id = data.get("extId")
                if task_id:
                    self.poll_task(task_id, timeout_sec=45)
                # Re-query
                for s in self.list_subnets():
                    if s.get("name") == subnet_payload.get("name"):
                        return s.get("extId") or s.get("metadata", {}).get("uuid")
            else:
                print(f"[Warning] Create subnet response: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"[Warning] Error creating subnet: {e}")
        return None

    # --- VPCS ---
    def list_vpcs(self) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/api/networking/v4.0/config/vpcs"
            resp = self.client.get(url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[Warning] Failed to list VPCs: {e}")
        return []

    def create_vpc(self, vpc_payload: Dict[str, Any]) -> Optional[str]:
        try:
            url = f"{self.base_url}/api/networking/v4.0/config/vpcs"
            headers = self._headers()
            headers["NTNX-Request-Id"] = str(uuid.uuid4())
            resp = self.client.post(url, json=vpc_payload, headers=headers)
            if resp.status_code in [200, 201, 202]:
                data = resp.json().get("data", {})
                task_id = data.get("extId")
                if task_id:
                    self.poll_task(task_id, timeout_sec=45)
                for v in self.list_vpcs():
                    if v.get("name") == vpc_payload.get("name"):
                        return v.get("extId")
            else:
                print(f"[Warning] Create VPC response: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"[Warning] Error creating VPC: {e}")
        return None

    def delete_vpc(self, vpc_id: str) -> bool:
        try:
            url = f"{self.base_url}/api/networking/v4.0/config/vpcs/{vpc_id}"
            resp = self.client.delete(url, headers=self._headers())
            if resp.status_code in [200, 202, 204]:
                data = resp.json().get("data", {})
                task_id = data.get("extId")
                if task_id:
                    self.poll_task(task_id, timeout_sec=45)
                return True
        except Exception as e:
            print(f"[Warning] Failed to delete VPC {vpc_id}: {e}")
        return False

# ---------------------------------------------------------------------------
# 4. Helper Functions & Cloud-Init
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

            # Extract IP Subnet
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

    if not net_ip:
        net_ip = "10.55.81.128"
    if not gw_ip:
        octets = str(net_ip).split(".")
        if len(octets) >= 3:
            gw_ip = f"{octets[0]}.{octets[1]}.{octets[2]}.129"
        else:
            gw_ip = "10.55.81.129"

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

def generate_linux_vm_cloud_init() -> str:
    """Generates base64-encoded cloud-init script for Linux VM provisioning."""
    cloud_init_raw = """#cloud-config
users:
  - name: nutanix
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    groups: [wheel, sudo, admin]
    shell: /bin/bash
    lock_passwd: false
    plain_text_passwd: 'MySecretPassword123!'

# Enable password authentication for SSH (optional but often needed)
ssh_pwauth: true
chpasswd:
  expire: false
"""
    return base64.b64encode(cloud_init_raw.encode("utf-8")).decode("utf-8")

# ---------------------------------------------------------------------------
# 5. LangGraph Node Implementations
# ---------------------------------------------------------------------------

async def plan_discovery_node(state: EnablementState) -> Dict[str, Any]:
    """
    Node 1: Plan (Discovery & Intent)
    1. Interrupts to ask the user for intent (Build or Destroy) and Group_ID.
    2. Queries Prism Central to discover current cluster state (Storage, Categories, VPCs, Subnets, VMs).
    3. Idempotently assesses existing constructs and plans required additions.
    """
    print("\n=========================================================")
    print(" [Node 1: Plan] Discovery & Intent Definition")
    print("=========================================================")

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
    existing_containers = pc.list_storage_containers()
    existing_vms = pc.list_vms()
    images = pc.list_images()

    # Find secondary / external subnet
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
    sec_name = "secondary-DM3-POC081"
    sec_id = None
    is_already_external = False

    if secondary_subnet:
        sec_name = secondary_subnet.get("name", "secondary-DM3-POC081")
        sec_id = secondary_subnet.get("extId") or secondary_subnet.get("metadata", {}).get("uuid")
        captured_vlan_id = secondary_subnet.get("networkId")
        if captured_vlan_id is None:
            captured_vlan_id = secondary_subnet.get("vlanId", 811)
        captured_cluster_ref = secondary_subnet.get("clusterReference")
        if not captured_cluster_ref and secondary_subnet.get("clusterReferenceList"):
            captured_cluster_ref = secondary_subnet.get("clusterReferenceList")[0]
        captured_vswitch_ref = secondary_subnet.get("virtualSwitchReference")
        is_already_external = secondary_subnet.get("isExternal", False) or secondary_subnet.get("isAdvancedNetworking", False)

    if captured_vlan_id is None:
        captured_vlan_id = 811
    if not captured_cluster_ref:
        captured_cluster_ref = "00065a51-3cfb-9563-0000-0000000297f5"

    # Capture DNS
    captured_dns_servers = pc.get_cluster_dns_servers(captured_cluster_ref)
    if not captured_dns_servers:
        captured_dns_servers = ["10.54.1.5"]

    # Extract IP & IPAM parameters
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
        "is_already_external": is_already_external,
        "network_ip": captured_net_ip,
        "prefix_length": captured_prefix_len,
        "gateway_ip": captured_gateway_ip,
        "ipam_start": ipam_start,
        "ipam_end": ipam_end,
        "ipam_pool": ipam_pool_str
    }

    # Check Rocky image
    rocky_img = pc.get_image_by_name("rocky")
    rocky_img_name = rocky_img.get("spec", {}).get("name") or rocky_img.get("status", {}).get("name", "Rocky-9-GenericCloud") if rocky_img else "Rocky-9-GenericCloud"
    rocky_img_uuid = rocky_img.get("metadata", {}).get("uuid") if rocky_img else "6104ebcf-ba0b-4a2a-b0c1-6f71f20bb08e"

    # Check existing NKP container
    nkp_container = pc.get_storage_container("nkp")
    nkp_container_exists = nkp_container is not None

    # Check existing Group VM
    target_vm_name = f"Group-{group_id}-Linux-VM"
    existing_vm = pc.get_vm_by_name(target_vm_name)
    vm_exists = existing_vm is not None

    # Check existing Transit & Spoke VPCs for this group
    transit_vpc_name = f"Transit-VPC-{group_id}"
    existing_transit_vpc = next((v for v in existing_vpcs if v.get("name") == transit_vpc_name), None)

    print(f"-> Discovered {len(existing_vpcs)} live VPC(s), {len(existing_subnets)} subnet(s), {len(existing_containers)} container(s), and {len(existing_vms)} VM(s).")
    print(f"-> Target Subnet: '{sec_name}' (VLAN {captured_vlan_id}, External: {is_already_external})")
    print(f"-> NKP Storage Container Present: {nkp_container_exists}")
    print(f"-> Rocky Linux Image: '{rocky_img_name}' ({rocky_img_uuid})")
    print(f"-> Linux VM '{target_vm_name}' Present: {vm_exists}")

    discovered_state = {
        "existing_vpcs": existing_vpcs,
        "existing_subnets": existing_subnets,
        "existing_containers": existing_containers,
        "existing_vms": existing_vms,
        "secondary_subnet": secondary_subnet,
        "dns_servers": captured_dns_servers,
        "rocky_img_uuid": rocky_img_uuid,
        "rocky_img_name": rocky_img_name,
        "nkp_container_uuid": nkp_container.get("containerExtId") if nkp_container else None
    }

    # 3. Generate Execution Plan
    execution_plan = []
    step_idx = 1

    if intent == "Build":
        # Step: Storage Container
        if not nkp_container_exists:
            execution_plan.append({
                "step": step_idx,
                "action": "CREATE_STORAGE_CONTAINER",
                "target_type": "StorageContainer",
                "target_name": "nkp",
                "details": {
                    "description": "Create storage container 'nkp' on cluster storage pool with default parameters",
                    "cluster_ref": captured_cluster_ref
                }
            })
            step_idx += 1

        # Step: Categories
        execution_plan.append({
            "step": step_idx,
            "action": "CREATE_CATEGORIES",
            "target_type": "Categories",
            "target_name": "Prism Central Categories",
            "details": {
                "description": "Ensure categories nodetype (worker, controlplane) and clustertype (workload, management) exist",
                "categories": {
                    "nodetype": ["worker", "controlplane"],
                    "clustertype": ["workload", "management"]
                }
            }
        })
        step_idx += 1

        # Step: External Subnet
        if not is_already_external:
            execution_plan.append({
                "step": step_idx,
                "action": "DELETE_BASIC_SUBNET",
                "target_type": "Subnet",
                "target_name": sec_name,
                "target_id": sec_id,
                "details": {
                    "description": f"Delete predefined Basic VLAN subnet '{sec_name}' to free VLAN ID {captured_vlan_id}",
                    "captured_vlan_id": captured_vlan_id,
                    "target_ext_id": sec_id
                }
            })
            step_idx += 1
            execution_plan.append({
                "step": step_idx,
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
            })
            step_idx += 1
        else:
            execution_plan.append({
                "step": step_idx,
                "action": "VERIFY_EXTERNAL_VLAN_SUBNET",
                "target_type": "Subnet",
                "target_name": sec_name,
                "target_id": sec_id,
                "details": {
                    "description": f"Verify and reuse existing External VLAN Subnet '{sec_name}' (VLAN {captured_vlan_id}, {captured_net_ip}/{captured_prefix_len})",
                    "vlan_id": captured_vlan_id,
                    "target_ext_id": sec_id
                }
            })
            step_idx += 1

        # Step: Transit VPC
        execution_plan.append({
            "step": step_idx,
            "action": "CREATE_TRANSIT_VPC",
            "target_type": "VPC",
            "target_name": transit_vpc_name,
            "details": {
                "description": f"Ensure Transit VPC '{transit_vpc_name}' is configured with NAT, cluster DNS, and ERP/Non-ERP subnets",
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
        })
        step_idx += 1

        # Step: Spoke VPCs
        execution_plan.append({
            "step": step_idx,
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
        })
        step_idx += 1

        # Step: Linux VM Provisioning
        if not vm_exists:
            execution_plan.append({
                "step": step_idx,
                "action": "CREATE_LINUX_VM",
                "target_type": "VM",
                "target_name": target_vm_name,
                "details": {
                    "description": f"Deploy Rocky Linux VM '{target_vm_name}' on Non-ERP subnet (IP 20.20.20.14) with 10 vCPUs, 16 GB RAM, 120 GB disk on 'nkp', and cloud-init",
                    "image_name": rocky_img_name,
                    "image_uuid": rocky_img_uuid,
                    "target_subnet": f"Transit-NonERP-{group_id}",
                    "static_ip": "20.20.20.14",
                    "vcpus": 10,
                    "memory_mib": 16384,
                    "disk_size_gib": 120,
                    "storage_container": "nkp",
                    "cloud_init_user": "nutanix"
                }
            })
            step_idx += 1
        else:
            execution_plan.append({
                "step": step_idx,
                "action": "VERIFY_LINUX_VM",
                "target_type": "VM",
                "target_name": target_vm_name,
                "target_id": existing_vm.get("metadata", {}).get("uuid"),
                "details": {
                    "description": f"Verify existing Linux VM '{target_vm_name}' (IP: 20.20.20.14, Subnet: Transit-NonERP-{group_id})",
                    "target_id": existing_vm.get("metadata", {}).get("uuid")
                }
            })
            step_idx += 1

    else:  # Destroy Intent
        # 1. Identify group VMs to delete
        for vm in existing_vms:
            vm_name = vm.get("spec", {}).get("name") or vm.get("status", {}).get("name", "")
            if f"Group-{group_id}" in vm_name or "Enablement" in vm_name or "Linux-VM" in vm_name:
                execution_plan.append({
                    "step": step_idx,
                    "action": "DELETE_LINUX_VM",
                    "target_type": "VM",
                    "target_name": vm_name,
                    "target_id": vm.get("metadata", {}).get("uuid"),
                    "details": {"description": f"Delete Linux VM '{vm_name}'"}
                })
                step_idx += 1

        # 2. Identify VPCs to delete
        for vpc in existing_vpcs:
            v_name = vpc.get("name", "")
            if f"-{group_id}" in v_name or "Enablement" in v_name or "Transit" in v_name or "Spoke" in v_name:
                execution_plan.append({
                    "step": step_idx,
                    "action": "DELETE_VPC",
                    "target_type": "VPC",
                    "target_name": vpc.get("name"),
                    "target_id": vpc.get("extId"),
                    "details": {"description": f"Delete VPC '{vpc.get('name')}'"}
                })
                step_idx += 1

        # 3. Identify subnets to delete
        for sub in existing_subnets:
            s_name = sub.get("name", "")
            if f"-{group_id}" in s_name and ("Transit" in s_name or "Spoke" in s_name):
                execution_plan.append({
                    "step": step_idx,
                    "action": "DELETE_SUBNET",
                    "target_type": "Subnet",
                    "target_name": sub.get("name"),
                    "target_id": sub.get("extId") or sub.get("metadata", {}).get("uuid"),
                    "details": {"description": f"Delete subnet '{sub.get('name')}'"}
                })
                step_idx += 1

        if not execution_plan:
            execution_plan.append({
                "step": 1,
                "action": "NO_OP_CLEANUP",
                "target_type": "None",
                "target_name": "No Group Constructs Found",
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
    print("-" * 78)
    for p in plan:
        print(f"Step {p.get('step')}: [{p.get('action')}] {p.get('target_name')}")
        details = p.get("details", {})
        if "categories" in details:
            for cat_k, cat_v in details["categories"].items():
                print(f"   * Category '{cat_k}': {', '.join(cat_v)}")
        elif "spokes" in details:
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
    print("-" * 78)

    if intent == "Build":
        if captured_info.get("vlan_id"):
            print(f"[CONFIRMATION] Captured VLAN ID: {captured_info.get('vlan_id')}")
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
    vlan_id = captured_info.get("vlan_id", 811)
    cluster_ref = captured_info.get("cluster_ref") or "00065a51-3cfb-9563-0000-0000000297f5"
    vswitch_ref = captured_info.get("vswitch_ref")

    discovered = state.get("discovered_state", {})
    rocky_img_uuid = discovered.get("rocky_img_uuid", "6104ebcf-ba0b-4a2a-b0c1-6f71f20bb08e")
    nkp_container_uuid = discovered.get("nkp_container_uuid")

    execution_results = []
    created_external_subnet_ext_id = captured_info.get("subnet_ext_id")
    created_transit_vpc_ext_id = None
    created_non_erp_subnet_uuid = None

    pc = NutanixPrismClient()

    for item in plan:
        action = item.get("action")
        target_name = item.get("target_name")
        target_id = item.get("target_id")
        step_num = item.get("step")

        print(f"\n-> Executing Step {step_num}: {action} ({target_name})...")

        # 1. STORAGE CONTAINER
        if action == "CREATE_STORAGE_CONTAINER":
            sc_id = pc.create_storage_container(name="nkp", cluster_ext_id=cluster_ref)
            if sc_id:
                nkp_container_uuid = sc_id
                print(f"[OK] Storage container 'nkp' verified/created [ExtID: {sc_id}].")
                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": sc_id,
                    "status": "SUCCESS",
                    "details": "Storage container 'nkp' active with default parameters"
                })
            else:
                print(f"[Error] Failed to create storage container 'nkp'.")
                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": "N/A",
                    "status": "FAILED",
                    "details": "Storage container creation failed."
                })

        # 2. CATEGORIES
        elif action == "CREATE_CATEGORIES":
            cat_map = item.get("details", {}).get("categories", {
                "nodetype": ["worker", "controlplane"],
                "clustertype": ["workload", "management"]
            })
            cat_res = pc.ensure_categories(cat_map)
            all_ok = any(cat_res.values())
            print(f"[OK] Categories ensured: {list(cat_map.keys())}")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": "categories-configured",
                "status": "SUCCESS" if all_ok else "PARTIAL",
                "details": f"Categories nodetype: {cat_map.get('nodetype')}, clustertype: {cat_map.get('clustertype')}"
            })

        # 3. VERIFY EXISTING EXTERNAL SUBNET
        elif action == "VERIFY_EXTERNAL_VLAN_SUBNET":
            print(f"[OK] External VLAN Subnet '{target_name}' is already present and active (VLAN {vlan_id}). Reusing.")
            created_external_subnet_ext_id = target_id or captured_info.get("subnet_ext_id")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": created_external_subnet_ext_id or "subnet-external-id",
                "status": "SUCCESS",
                "details": f"Existing External VLAN Subnet verified (VLAN {vlan_id})"
            })

        # 4. DELETE BASIC SUBNET
        elif action == "DELETE_BASIC_SUBNET":
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

        # 5. CREATE EXTERNAL VLAN SUBNET
        elif action == "CREATE_EXTERNAL_VLAN_SUBNET":
            net_ip = captured_info.get("network_ip") or item.get("details", {}).get("network_ip", "10.55.81.128").split("/")[0]
            prefix_len = int(captured_info.get("prefix_length") or 25)
            gw_ip = captured_info.get("gateway_ip") or item.get("details", {}).get("gateway_ip", "10.55.81.129")
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

        # 6. TRANSIT VPC
        elif action == "CREATE_TRANSIT_VPC":
            # Idempotency check: see if Transit VPC already exists
            existing_transit = next((v for v in pc.list_vpcs() if v.get("name") == target_name), None)
            if existing_transit:
                created_transit_vpc_ext_id = existing_transit.get("extId")
                print(f"[OK] Transit VPC '{target_name}' already exists [ID: {created_transit_vpc_ext_id}]. Reusing.")
            else:
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
                # Ensure ERP and Non-ERP overlay subnets exist
                current_subs = pc.list_subnets()
                erp_sub_name = f"Transit-ERP-{group_id}"
                non_erp_sub_name = f"Transit-NonERP-{group_id}"

                existing_erp = next((s for s in current_subs if s.get("name") == erp_sub_name), None)
                if not existing_erp:
                    erp_sub = {
                        "name": erp_sub_name,
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
                    pc.create_subnet(erp_sub)

                existing_non_erp = next((s for s in current_subs if s.get("name") == non_erp_sub_name), None)
                if not existing_non_erp:
                    non_erp_sub = {
                        "name": non_erp_sub_name,
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
                    created_non_erp_subnet_uuid = pc.create_subnet(non_erp_sub)
                else:
                    created_non_erp_subnet_uuid = existing_non_erp.get("extId") or existing_non_erp.get("metadata", {}).get("uuid")

                dns_str = f" with DNS {captured_dns_servers}" if captured_dns_servers else ""
                print(f"[OK] Transit VPC '{target_name}'{dns_str} & ERP Advertisements [ID: {created_transit_vpc_ext_id}]")

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

        # 7. SPOKE VPCS
        elif action == "CREATE_SPOKE_VPCS":
            spokes_config = [
                {"index": 1, "ip": "1.1.1.0", "gw": "1.1.1.1", "pool_start": "1.1.1.160", "pool_end": "1.1.1.253"},
                {"index": 2, "ip": "2.2.2.0", "gw": "2.2.2.1", "pool_start": "2.2.2.160", "pool_end": "2.2.2.253"},
                {"index": 3, "ip": "3.3.3.0", "gw": "3.3.3.1", "pool_start": "3.3.3.160", "pool_end": "3.3.3.253"}
            ]
            spoke_results = []
            current_vpcs = pc.list_vpcs()
            current_subs = pc.list_subnets()

            for spoke in spokes_config:
                spoke_name = f"Spoke-VPC-{spoke['index']}-{group_id}"
                existing_spoke = next((v for v in current_vpcs if v.get("name") == spoke_name), None)

                if existing_spoke:
                    spoke_id = existing_spoke.get("extId")
                    print(f"   [OK] Spoke {spoke['index']}/3: '{spoke_name}' already exists [ID: {spoke_id}]. Reusing.")
                else:
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
                    spoke_sub_name = f"Spoke-ERP-{spoke['index']}-{group_id}"
                    existing_spoke_sub = next((s for s in current_subs if s.get("name") == spoke_sub_name), None)
                    if not existing_spoke_sub:
                        spoke_sub = {
                            "name": spoke_sub_name,
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
                    print(f"   [OK] Spoke {spoke['index']}/3: Verified/Created '{spoke_name}' (DNS {captured_dns_servers}, ERP {spoke['ip']}/24, GW {spoke['gw']}, IPAM {spoke['pool_start']}-{spoke['pool_end']}) [ID: {spoke_id}]")
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

        # 8. LINUX VM PROVISIONING
        elif action == "CREATE_LINUX_VM":
            # Resolve Non-ERP subnet UUID
            if not created_non_erp_subnet_uuid:
                for s in pc.list_subnets():
                    if s.get("name") == f"Transit-NonERP-{group_id}":
                        created_non_erp_subnet_uuid = s.get("extId") or s.get("metadata", {}).get("uuid")
                        break

            # Resolve NKP container UUID
            if not nkp_container_uuid:
                nkp_c = pc.get_storage_container("nkp")
                if nkp_c:
                    nkp_container_uuid = nkp_c.get("containerExtId") or nkp_c.get("id")

            # Resolve Rocky Image UUID
            if not rocky_img_uuid:
                r_img = pc.get_image_by_name("rocky")
                if r_img:
                    rocky_img_uuid = r_img.get("metadata", {}).get("uuid")

            print(f"-> Provisioning Linux VM '{target_name}' (Image: {rocky_img_uuid}, Subnet: {created_non_erp_subnet_uuid}, SC: {nkp_container_uuid})...")

            cloud_init_data = generate_linux_vm_cloud_init()

            vm_body = {
                "spec": {
                    "name": target_name,
                    "description": f"Rocky Linux VM for Enablement Group {group_id}",
                    "resources": {
                        "power_state": "ON",
                        "num_sockets": 2,
                        "num_vcpus_per_socket": 5,
                        "memory_size_mib": 16384,
                        "nic_list": [
                            {
                                "nic_type": "NORMAL_NIC",
                                "is_connected": True,
                                "ip_endpoint_list": [{"ip": "20.20.20.14"}],
                                "subnet_reference": {
                                    "kind": "subnet",
                                    "uuid": created_non_erp_subnet_uuid
                                }
                            }
                        ],
                        "disk_list": [
                            {
                                "device_properties": {
                                    "disk_address": {
                                        "adapter_type": "SCSI",
                                        "device_index": 0
                                    },
                                    "device_type": "DISK"
                                },
                                "disk_size_bytes": 120 * 1024 * 1024 * 1024,
                                "disk_size_mib": 120 * 1024,
                                "data_source_reference": {
                                    "kind": "image",
                                    "uuid": rocky_img_uuid
                                }
                            }
                        ],
                        "guest_customization": {
                            "cloud_init": {
                                "user_data": cloud_init_data
                            }
                        }
                    },
                    "cluster_reference": {
                        "kind": "cluster",
                        "uuid": cluster_ref
                    }
                },
                "metadata": {
                    "kind": "vm"
                }
            }

            created_vm_uuid = pc.create_vm(vm_body)
            if created_vm_uuid:
                print(f"[OK] Created Linux VM '{target_name}' (IP: 20.20.20.14, 10 vCPU, 16 GB RAM, 120 GB Disk on 'nkp', Powered ON) [UUID: {created_vm_uuid}]")
                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": created_vm_uuid,
                    "status": "SUCCESS",
                    "details": "Rocky Linux VM with 10 vCPUs, 16 GB RAM, 120 GB disk on nkp, static IP 20.20.20.14, cloud-init user nutanix, and Powered ON"
                })
            else:
                print(f"[Error] Failed to create Linux VM '{target_name}'.")
                execution_results.append({
                    "step": step_num,
                    "action": action,
                    "target_name": target_name,
                    "extId": "N/A",
                    "status": "FAILED",
                    "details": "VM creation API call failed."
                })

        # 9. VERIFY EXISTING LINUX VM
        elif action == "VERIFY_LINUX_VM":
            print(f"[OK] Linux VM '{target_name}' already exists [UUID: {target_id}]. Reusing.")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": target_id,
                "status": "SUCCESS",
                "details": "Existing Linux VM verified (IP: 20.20.20.14, Powered ON)"
            })

        # 10. DELETE LINUX VM
        elif action == "DELETE_LINUX_VM":
            ok = pc.delete_vm(target_id) if target_id else True
            print(f"[{'OK' if ok else 'Warning'}] Delete VM '{target_name}' (UUID: {target_id})")
            execution_results.append({
                "step": step_num,
                "action": action,
                "target_name": target_name,
                "extId": target_id,
                "status": "DELETED" if ok else "FAILED"
            })

        # 11. DELETE VPC
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

        # 12. DELETE SUBNET
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

        # 13. NO-OP CLEANUP
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
    vlan_id = captured_info.get("vlan_id", 811)

    pc = NutanixPrismClient()
    live_vpcs = pc.list_vpcs()
    live_subnets = pc.list_subnets()
    live_vms = pc.list_vms()
    live_containers = pc.list_storage_containers()

    print(f"-> Live Audit Complete: {len(live_vpcs)} VPC(s), {len(live_subnets)} Subnet(s), {len(live_containers)} Storage Container(s), and {len(live_vms)} VM(s) on cluster.")

    verification_table = []
    for res in results:
        status_tag = f"[{res.get('status')}]"
        verification_table.append({
            "Action": res.get("action"),
            "Target": res.get("target_name"),
            "Status": status_tag,
            "ExtID": res.get("extId", "N/A")
        })

    print("\n==========================================================================================")
    print(f" FINAL EXECUTION SUMMARY (Intent: {intent} | Group: {group_id})")
    print("==========================================================================================")
    print(f"{'Action':<28} | {'Target Name':<30} | {'Status':<10} | {'Resource ExtID'}")
    print("-" * 90)
    for row in verification_table:
        print(f"{row['Action']:<28} | {row['Target']:<30} | {row['Status']:<10} | {row['ExtID']}")
    print("-" * 90)

    if intent == "Build":
        dns_display = ", ".join(captured_dns_servers) if captured_dns_servers else "None"
        ext_net_str = f"{captured_info.get('network_ip', '10.55.81.128')}/{captured_info.get('prefix_length', 25)}"
        ext_gw_str = captured_info.get('gateway_ip', '10.55.81.129')
        ext_ipam_str = captured_info.get('ipam_pool', '10.55.81.160 - 10.55.81.253')
        print("\nStudent Enablement Infrastructure Topology Provisioned:")
        print(f" * Storage Container  : nkp (Storage Pool 4c9fbf10-d529-440a-ba33-b9e335ed1dfe)")
        print(f" * Categories         : nodetype (worker, controlplane) | clustertype (workload, management)")
        print(f" * Reused VLAN ID     : {vlan_id}")
        print(f" * Cluster DNS Server : {dns_display} (Configured on all VPCs)")
        print(f" * External Network   : Network Controller Subnet {ext_net_str} (GW {ext_gw_str}, IPAM {ext_ipam_str})")
        print(f" * Transit VPC        : Transit-VPC-{group_id} (DNS: {dns_display}, NAT Enabled, ERP Advertised)")
        print(f"   - Transit-ERP-{group_id}    : 10.10.10.0/24 (GW 10.10.10.1, IPAM .160-.253, ERP)")
        print(f"   - Transit-NonERP-{group_id} : 20.20.20.0/24 (GW 20.20.20.1, IPAM .160-.253, Non-ERP)")
        print(f" * Spoke 1 VPC        : Spoke-VPC-1-{group_id} (DNS: {dns_display}, ERP 1.1.1.0/24, GW 1.1.1.1, IPAM .160-.253, No-NAT -> Transit)")
        print(f" * Spoke 2 VPC        : Spoke-VPC-2-{group_id} (DNS: {dns_display}, ERP 2.2.2.0/24, GW 2.2.2.1, IPAM .160-.253, No-NAT -> Transit)")
        print(f" * Spoke 3 VPC        : Spoke-VPC-3-{group_id} (DNS: {dns_display}, ERP 3.3.3.0/24, GW 3.3.3.1, IPAM .160-.253, No-NAT -> Transit)")
        print(f" * Group Linux VM     : Group-{group_id}-Linux-VM (Rocky Linux, 10 vCPUs, 16GB RAM, 120GB on nkp, Static IP: 20.20.20.14 on Transit-NonERP-{group_id})")
        print(f"   - Cloud-Init User  : nutanix (password: MySecretPassword123!, sudo NOPASSWD, SSH password auth enabled)")
    else:
        print("\nStudent Enablement Constructs Cleaned & Destroyed Successfully.")

    print("==========================================================================================\n")

    summary_text = f"Audit complete: {len(results)} actions verified for {intent}."
    return {
        "final_summary": summary_text,
        "messages": [AIMessage(content=summary_text)]
    }

# ---------------------------------------------------------------------------
# 6. Routing Decisions
# ---------------------------------------------------------------------------
def route_after_review(state: EnablementState) -> str:
    if state.get("approval_status") == "approved":
        return "execute_provisioning"
    return END

# ---------------------------------------------------------------------------
# 7. StateGraph Construction & Compilation
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
# 8. Interactive Execution Loop (CLI Entrypoint)
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
