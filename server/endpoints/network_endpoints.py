"""Network Management Endpoints for Thoth Device.

This module handles network configuration including:
- WiFi setup via captive portal
- Network status monitoring
- Connection management
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import subprocess
import re

# Import shared models
from .models import StandardResponse

router = APIRouter(prefix="/network", tags=["network"])

# ============================================================================
# MODELS
# ============================================================================

class WiFiConfig(BaseModel):
    """WiFi configuration parameters."""
    ssid: str = Field(..., description="WiFi network name")
    password: str = Field(..., description="WiFi password", min_length=8)
    security: str = Field("WPA2", description="Security protocol (WPA2, WPA3, Open)")
    auto_connect: bool = Field(True, description="Auto-connect on boot")
    hidden: bool = Field(False, description="Hidden network")

class NetworkStatus(BaseModel):
    """Current network status."""
    connected: bool
    ssid: Optional[str] = None
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    signal_strength: Optional[int] = None  # dBm
    link_speed: Optional[int] = None  # Mbps
    gateway: Optional[str] = None
    dns_servers: List[str] = []

class NetworkScan(BaseModel):
    """Available WiFi networks."""
    ssid: str
    bssid: str
    signal_strength: int
    security: str
    frequency: int
    channel: int

class CaptivePortalConfig(BaseModel):
    """Captive portal configuration."""
    enabled: bool = Field(True, description="Enable captive portal")
    portal_name: str = Field("Thoth Setup", description="Portal display name")
    timeout: int = Field(300, description="Auto-disable after seconds")
    redirect_url: Optional[str] = Field(None, description="Redirect after setup")

# ============================================================================
# IN-MEMORY STORAGE (Replace with persistent storage in production)
# ============================================================================

# Store WiFi configurations
wifi_configs: Dict[str, WiFiConfig] = {}

# Store network status
network_status_cache: Dict[str, NetworkStatus] = {}

# Captive portal state
captive_portal_active = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_network_interface(interface: str = "wlan0") -> Dict[str, Any]:
    """Get network interface information.
    
    In production, this would use system calls to get actual network info.
    """
    try:
        # Mock data for development
        # In production, use subprocess to run iwconfig, ifconfig, etc.
        return {
            "interface": interface,
            "status": "connected",
            "ip": "192.168.1.100",
            "mac": "b8:27:eb:12:34:56",
            "ssid": "ThothNetwork",
            "signal": -45,
            "speed": 72
        }
    except Exception as e:
        return {"error": str(e)}

def scan_wifi_networks() -> List[Dict[str, Any]]:
    """Scan for available WiFi networks.
    
    In production, this would use iwlist or nmcli to scan networks.
    """
    # Mock data for development
    return [
        {
            "ssid": "ThothNetwork",
            "bssid": "00:11:22:33:44:55",
            "signal": -45,
            "security": "WPA2",
            "frequency": 2437,
            "channel": 6
        },
        {
            "ssid": "Guest_Network",
            "bssid": "00:11:22:33:44:66",
            "signal": -60,
            "security": "Open",
            "frequency": 2412,
            "channel": 1
        },
        {
            "ssid": "Research_Lab",
            "bssid": "00:11:22:33:44:77",
            "signal": -70,
            "security": "WPA3",
            "frequency": 5180,
            "channel": 36
        }
    ]

def configure_wifi_connection(config: WiFiConfig) -> bool:
    """Configure WiFi connection using system tools.
    
    In production, this would use nmcli or wpa_supplicant to configure WiFi.
    """
    try:
        # Mock implementation
        # In production, would execute:
        # nmcli dev wifi connect "{ssid}" password "{password}"
        # or update wpa_supplicant.conf
        
        wifi_configs[config.ssid] = config
        return True
    except Exception:
        return False

def start_captive_portal(config: CaptivePortalConfig) -> bool:
    """Start captive portal for WiFi configuration.
    
    In production, this would:
    1. Start hostapd for AP mode
    2. Configure dnsmasq for DHCP
    3. Set up iptables for captive portal redirect
    """
    global captive_portal_active
    try:
        # Mock implementation
        captive_portal_active = config.enabled
        return True
    except Exception:
        return False

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/network/wifi", response_model=StandardResponse)
async def configure_wifi(config: WiFiConfig):
    """Configure WiFi connection via captive portal or API.
    
    Sets up WiFi credentials and connects to the specified network.
    Supports WPA2, WPA3, and open networks.
    """
    try:
        # Validate SSID format
        if not config.ssid or len(config.ssid) > 32:
            raise ValueError("Invalid SSID length (1-32 characters)")
        
        # Validate password for secured networks
        if config.security != "Open" and len(config.password) < 8:
            raise ValueError("Password must be at least 8 characters for secured networks")
        
        # Configure WiFi
        success = configure_wifi_connection(config)
        
        if not success:
            raise Exception("Failed to configure WiFi connection")
        
        # Store configuration
        wifi_configs[config.ssid] = config
        
        return StandardResponse(
            success=True,
            message=f"WiFi configured for network: {config.ssid}",
            data={
                "ssid": config.ssid,
                "auto_connect": config.auto_connect,
                "security": config.security
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WiFi configuration failed: {str(e)}")

@router.get("/network/status", response_model=NetworkStatus)
async def get_network_status(interface: str = Query("wlan0", description="Network interface")):
    """Get current network connection status.
    
    Returns detailed information about the active network connection including:
    - Connection state
    - IP address
    - Signal strength
    - Link speed
    """
    try:
        # Get interface information
        info = get_network_interface(interface)
        
        if "error" in info:
            raise Exception(info["error"])
        
        # Build status response
        status = NetworkStatus(
            connected=info.get("status") == "connected",
            ssid=info.get("ssid"),
            ip_address=info.get("ip"),
            mac_address=info.get("mac"),
            signal_strength=info.get("signal"),
            link_speed=info.get("speed"),
            gateway="192.168.1.1",  # Mock gateway
            dns_servers=["8.8.8.8", "8.8.4.4"]  # Mock DNS
        )
        
        # Cache status
        network_status_cache[interface] = status
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get network status: {str(e)}")

@router.get("/network/scan", response_model=List[NetworkScan])
async def scan_networks():
    """Scan for available WiFi networks.
    
    Returns a list of detected networks with signal strength and security info.
    """
    try:
        # Scan for networks
        networks = scan_wifi_networks()
        
        # Convert to response models
        scan_results = [
            NetworkScan(
                ssid=net["ssid"],
                bssid=net["bssid"],
                signal_strength=net["signal"],
                security=net["security"],
                frequency=net["frequency"],
                channel=net["channel"]
            )
            for net in networks
        ]
        
        # Sort by signal strength (strongest first)
        scan_results.sort(key=lambda x: x.signal_strength, reverse=True)
        
        return scan_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Network scan failed: {str(e)}")

@router.post("/network/captive-portal", response_model=StandardResponse)
async def manage_captive_portal(config: CaptivePortalConfig):
    """Start or stop the captive portal for WiFi configuration.
    
    The captive portal provides a web interface for configuring WiFi
    when no network connection is available.
    """
    try:
        if config.enabled:
            # Start captive portal
            success = start_captive_portal(config)
            if success:
                message = f"Captive portal started: {config.portal_name}"
            else:
                raise Exception("Failed to start captive portal")
        else:
            # Stop captive portal
            global captive_portal_active
            captive_portal_active = False
            message = "Captive portal stopped"
        
        return StandardResponse(
            success=True,
            message=message,
            data={
                "portal_active": captive_portal_active,
                "portal_name": config.portal_name if config.enabled else None,
                "timeout": config.timeout if config.enabled else None
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Captive portal operation failed: {str(e)}")

@router.get("/network/saved", response_model=Dict[str, Any])
async def get_saved_networks():
    """Get list of saved WiFi networks.
    
    Returns all configured networks with their settings (passwords excluded).
    """
    try:
        saved = []
        for ssid, config in wifi_configs.items():
            saved.append({
                "ssid": ssid,
                "security": config.security,
                "auto_connect": config.auto_connect,
                "hidden": config.hidden,
                "last_connected": datetime.now().isoformat()  # Mock timestamp
            })
        
        return {
            "success": True,
            "count": len(saved),
            "networks": saved
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve saved networks: {str(e)}")

@router.delete("/network/wifi/{ssid}", response_model=StandardResponse)
async def forget_network(ssid: str):
    """Remove a saved WiFi network configuration.
    
    Forgets the network and removes stored credentials.
    """
    try:
        if ssid not in wifi_configs:
            raise HTTPException(status_code=404, detail=f"Network '{ssid}' not found")
        
        # Remove configuration
        del wifi_configs[ssid]
        
        # In production, also remove from system configuration
        # nmcli connection delete id "{ssid}"
        
        return StandardResponse(
            success=True,
            message=f"Network '{ssid}' has been forgotten",
            data={"ssid": ssid}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to forget network: {str(e)}")

@router.post("/network/reconnect", response_model=StandardResponse)
async def reconnect_network(ssid: Optional[str] = None):
    """Reconnect to a WiFi network.
    
    If SSID is provided, connects to that specific network.
    Otherwise, attempts to connect to the best available saved network.
    """
    try:
        if ssid:
            if ssid not in wifi_configs:
                raise HTTPException(status_code=404, detail=f"Network '{ssid}' not configured")
            
            config = wifi_configs[ssid]
            success = configure_wifi_connection(config)
            
            if success:
                message = f"Reconnected to {ssid}"
            else:
                raise Exception(f"Failed to connect to {ssid}")
        else:
            # Try to connect to any saved network
            if not wifi_configs:
                raise HTTPException(status_code=404, detail="No saved networks available")
            
            # Try auto-connect networks first
            for ssid, config in wifi_configs.items():
                if config.auto_connect:
                    if configure_wifi_connection(config):
                        message = f"Auto-connected to {ssid}"
                        break
            else:
                raise Exception("No networks available for auto-connect")
        
        return StandardResponse(
            success=True,
            message=message,
            data={"connected_to": ssid}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconnection failed: {str(e)}")
