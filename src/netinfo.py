"""Hostname / IP helpers used to tell people where the web interface is."""
import socket
from typing import List, Optional


def hostname() -> str:
    return socket.gethostname().split(".")[0]


def local_ip() -> Optional[str]:
    """The LAN address of this machine, or None when offline."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent
            ip = s.getsockname()[0]
            return None if ip.startswith("127.") else ip
    except OSError:
        return None


def web_urls(port: int, host: str = "0.0.0.0") -> List[str]:
    """URLs a person can type to reach the web interface, best first."""
    if host not in ("0.0.0.0", "", "::"):
        return [f"http://{host}:{port}"]
    urls = [f"http://{hostname()}.local:{port}"]  # mDNS via avahi (set up by setup_pi.sh)
    ip = local_ip()
    if ip:
        urls.append(f"http://{ip}:{port}")
    return urls
