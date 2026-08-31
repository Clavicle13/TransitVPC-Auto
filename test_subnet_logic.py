import unittest
from typing import Dict, Any
from main import extract_subnet_ip_details

class TestSubnetIPLogic(unittest.TestCase):

    def test_standard_v4_nutanix_subnet(self):
        """Test extraction from standard Nutanix v4 Networking API response."""
        mock_subnet = {
            "name": "secondary-POC227",
            "extId": "sub-12345",
            "networkId": 2271,
            "ipConfig": [
                {
                    "ipv4": {
                        "ipSubnet": {
                            "ip": {"value": "10.55.81.0", "prefixLength": 32},
                            "prefixLength": 24
                        },
                        "defaultGatewayIp": {"value": "10.55.81.1", "prefixLength": 32},
                        "poolList": [
                            {
                                "startIp": {"value": "10.55.81.50", "prefixLength": 32},
                                "endIp": {"value": "10.55.81.100", "prefixLength": 32}
                            }
                        ]
                    }
                }
            ]
        }

        result = extract_subnet_ip_details(mock_subnet)
        self.assertEqual(result["network_ip"], "10.55.81.0")
        self.assertEqual(result["prefix_length"], 24)
        self.assertEqual(result["gateway_ip"], "10.55.81.1")
        self.assertEqual(result["ipam_start"], "10.55.81.160")
        self.assertEqual(result["ipam_end"], "10.55.81.253")
        self.assertEqual(result["ipam_pool"], "10.55.81.160 - 10.55.81.253")

    def test_non_standard_subnet_mask(self):
        """Test with /25 network and gateway at .129."""
        mock_subnet = {
            "name": "secondary-custom",
            "ipConfig": [
                {
                    "ipv4": {
                        "ipSubnet": {
                            "ip": {"value": "10.136.227.128"},
                            "prefixLength": 25
                        },
                        "defaultGatewayIp": {"value": "10.136.227.129"}
                    }
                }
            ]
        }

        result = extract_subnet_ip_details(mock_subnet)
        self.assertEqual(result["network_ip"], "10.136.227.128")
        self.assertEqual(result["prefix_length"], 25)
        self.assertEqual(result["gateway_ip"], "10.136.227.129")
        self.assertEqual(result["ipam_start"], "10.136.227.160")
        self.assertEqual(result["ipam_end"], "10.136.227.253")

    def test_string_format_ip_subnet(self):
        """Test with CIDR string format in ipSubnet."""
        mock_subnet = {
            "name": "secondary-string-format",
            "ipConfig": [
                {
                    "ipv4": {
                        "ipSubnet": "192.168.50.0/24",
                        "defaultGatewayIp": "192.168.50.1"
                    }
                }
            ]
        }

        result = extract_subnet_ip_details(mock_subnet)
        self.assertEqual(result["network_ip"], "192.168.50.0")
        self.assertEqual(result["prefix_length"], 24)
        self.assertEqual(result["gateway_ip"], "192.168.50.1")
        self.assertEqual(result["ipam_start"], "192.168.50.160")
        self.assertEqual(result["ipam_end"], "192.168.50.253")

    def test_minimal_or_empty_subnet(self):
        """Test fallback when subnet has no IPAM configured initially."""
        mock_subnet = {
            "name": "secondary-empty",
            "networkId": 100
        }

        result = extract_subnet_ip_details(mock_subnet)
        self.assertTrue(result["ipam_start"].endswith(".160"))
        self.assertTrue(result["ipam_end"].endswith(".253"))

if __name__ == "__main__":
    unittest.main()
