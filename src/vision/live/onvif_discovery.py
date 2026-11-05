"""User-initiated ONVIF WS-Discovery on one explicitly selected local IPv4 interface."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address
from socket import AF_INET, IPPROTO_IP, IP_MULTICAST_IF, SOCK_DGRAM, inet_aton, socket
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4
from xml.etree import ElementTree


@dataclass(frozen=True)
class DiscoveredCamera:
    address: str
    onvif_endpoint: str
    name: str = "ONVIF camera"
    brand: str = "Unknown"
    model: str = "Unknown"
    stream_available: bool | None = None


def parse_probe_response(payload: bytes) -> list[DiscoveredCamera]:
    """Extract advertised ONVIF endpoints; discovery alone never proves stream access."""
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return []
    devices = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "XAddrs" or not element.text:
            continue
        for address in element.text.split():
            parsed = urlsplit(address)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
                continue
            devices.append(DiscoveredCamera(parsed.hostname, address))
    return devices


def discover_onvif(interface_ipv4: str, *, timeout_s: float = 2.0, socket_factory=socket) -> list[DiscoveredCamera]:
    """Multicast only through the named local interface after explicit invocation."""
    try:
        IPv4Address(interface_ipv4)
    except ValueError as exc:
        raise ValueError("Choose a local IPv4 interface address for discovery.") from exc
    if timeout_s <= 0 or timeout_s > 10:
        raise ValueError("Discovery timeout must be between 0 and 10 seconds.")
    probe = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">'
        '<e:Header><w:MessageID>uuid:' + str(uuid4()) + '</w:MessageID>'
        '<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
        '</e:Header><e:Body><d:Probe/></e:Body></e:Envelope>'
    ).encode()
    connection = socket_factory(AF_INET, SOCK_DGRAM)
    found: dict[str, DiscoveredCamera] = {}
    try:
        connection.setsockopt(IPPROTO_IP, IP_MULTICAST_IF, inet_aton(interface_ipv4))
        connection.bind((interface_ipv4, 0))
        connection.sendto(probe, ("239.255.255.250", 3702))
        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            connection.settimeout(max(.01, deadline - monotonic()))
            try:
                payload, _ = connection.recvfrom(65535)
            except TimeoutError:
                break
            for camera in parse_probe_response(payload):
                found[camera.onvif_endpoint] = camera
    finally:
        connection.close()
    return list(found.values())
