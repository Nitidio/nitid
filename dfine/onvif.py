"""Minimal ONVIF discovery and Media1 client for camera stream resolution."""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
WSA_NS = "http://www.w3.org/2005/08/addressing"
WSD_NS = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
WSSE_NS = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU_NS = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
DEVICE_NS = "http://www.onvif.org/ver10/device/wsdl"
MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"
SCHEMA_NS = "http://www.onvif.org/ver10/schema"
DISCOVERY_ADDRESS = ("239.255.255.250", 3702)

for prefix, namespace in {
    "s": SOAP_NS,
    "a": WSA_NS,
    "d": WSD_NS,
    "wsse": WSSE_NS,
    "wsu": WSU_NS,
    "tds": DEVICE_NS,
    "trt": MEDIA_NS,
    "tt": SCHEMA_NS,
}.items():
    ET.register_namespace(prefix, namespace)


class ONVIFError(RuntimeError):
    """Raised for discovery, transport, authentication, and SOAP protocol errors."""


@dataclass(frozen=True, slots=True)
class ONVIFDevice:
    """One device returned by WS-Discovery."""

    endpoint_reference: str | None
    xaddrs: tuple[str, ...]
    scopes: tuple[str, ...] = ()
    types: tuple[str, ...] = ()

    @property
    def service_url(self) -> str | None:
        """Preferred ONVIF device-service endpoint."""
        return next(
            (address for address in self.xaddrs if address.startswith("https://")), None
        ) or (self.xaddrs[0] if self.xaddrs else None)


@dataclass(frozen=True, slots=True)
class ONVIFMediaProfile:
    """Relevant video settings from an ONVIF Media1 profile."""

    token: str
    name: str
    encoding: str | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None

    @property
    def resolution(self) -> tuple[int, int] | None:
        if self.width is None or self.height is None:
            return None
        return self.width, self.height


class SOAPTransport(Protocol):
    def request(self, url: str, action: str, payload: bytes, timeout: float) -> bytes: ...


class UrllibSOAPTransport:
    """SOAP-over-HTTP transport supporting Basic and Digest authentication."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        verify_ssl: bool = True,
    ) -> None:
        self.username = username
        self.password = password
        self.verify_ssl = verify_ssl

    def request(self, url: str, action: str, payload: bytes, timeout: float) -> bytes:
        handlers: list[Any] = []
        if self.username is not None:
            password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_manager.add_password(None, url, self.username, self.password or "")
            handlers.extend(
                [
                    urllib.request.HTTPDigestAuthHandler(password_manager),
                    urllib.request.HTTPBasicAuthHandler(password_manager),
                ]
            )
        if url.lower().startswith("https://"):
            context = ssl.create_default_context()
            if not self.verify_ssl:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=context))
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"',
                "User-Agent": "nitid-onvif/0.1",
            },
            method="POST",
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ONVIFError(f"ONVIF HTTP {exc.code} from {url}: {_fault_text(detail)}") from exc
        except urllib.error.URLError as exc:
            raise ONVIFError(f"Failed to reach ONVIF service at {url}: {exc.reason}") from exc


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element.iter() if _local_name(item.tag) == name), None)


def _first_text(element: ET.Element, name: str) -> str | None:
    match = _first_descendant(element, name)
    if match is None or match.text is None:
        return None
    value = match.text.strip()
    return value or None


def _split_text(element: ET.Element, name: str) -> tuple[str, ...]:
    text = _first_text(element, name)
    return tuple(text.split()) if text else ()


def _parse_xml(payload: bytes | str) -> ET.Element:
    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ONVIFError(f"Invalid ONVIF XML response: {exc}") from exc


def _fault_text(payload: bytes | str) -> str:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
        return text.strip()[:300] or "empty response"
    reason = _first_text(root, "Text") or _first_text(root, "Reason")
    return reason or "SOAP fault"


def build_discovery_probe(message_id: str | None = None) -> bytes:
    """Build a WS-Discovery Probe without filters for broad device compatibility."""
    envelope = ET.Element(f"{{{SOAP_NS}}}Envelope")
    header = ET.SubElement(envelope, f"{{{SOAP_NS}}}Header")
    ET.SubElement(header, f"{{{WSA_NS}}}Action").text = f"{WSD_NS}/Probe"
    ET.SubElement(header, f"{{{WSA_NS}}}MessageID").text = message_id or f"urn:uuid:{uuid.uuid4()}"
    reply_to = ET.SubElement(header, f"{{{WSA_NS}}}ReplyTo")
    ET.SubElement(
        reply_to, f"{{{WSA_NS}}}Address"
    ).text = "http://www.w3.org/2005/08/addressing/anonymous"
    ET.SubElement(header, f"{{{WSA_NS}}}To").text = "urn:schemas-xmlsoap-org:ws:2005:04:discovery"
    body = ET.SubElement(envelope, f"{{{SOAP_NS}}}Body")
    ET.SubElement(body, f"{{{WSD_NS}}}Probe")
    return ET.tostring(envelope, encoding="utf-8", xml_declaration=True)


def parse_discovery_response(payload: bytes | str) -> list[ONVIFDevice]:
    """Parse all ProbeMatch entries from one WS-Discovery datagram."""
    root = _parse_xml(payload)
    devices: list[ONVIFDevice] = []
    for match in (item for item in root.iter() if _local_name(item.tag) == "ProbeMatch"):
        xaddrs = _split_text(match, "XAddrs")
        if not xaddrs:
            continue
        devices.append(
            ONVIFDevice(
                endpoint_reference=_first_text(match, "Address"),
                xaddrs=xaddrs,
                scopes=_split_text(match, "Scopes"),
                types=_split_text(match, "Types"),
            )
        )
    return devices


def discover_onvif_devices(
    timeout: float = 3.0,
    *,
    interface: str | None = None,
    _socket_factory: Callable[..., Any] = socket.socket,
    _clock: Callable[[], float] = time.monotonic,
) -> list[ONVIFDevice]:
    """Discover ONVIF devices using IPv4 WS-Discovery multicast."""
    if timeout <= 0:
        raise ValueError("discovery timeout must be > 0")
    probe = build_discovery_probe()
    discovered: dict[str, ONVIFDevice] = {}
    sock = _socket_factory(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        if interface is not None:
            try:
                interface_bytes = socket.inet_aton(interface)
            except OSError as exc:
                raise ValueError(f"invalid IPv4 discovery interface: '{interface}'") from exc
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, interface_bytes)
        try:
            sock.sendto(probe, DISCOVERY_ADDRESS)
        except OSError as exc:
            raise ONVIFError(f"Failed to send ONVIF discovery probe: {exc}") from exc
        deadline = _clock() + timeout
        while True:
            remaining = deadline - _clock()
            if remaining <= 0:
                break
            sock.settimeout(min(remaining, 0.25))
            try:
                payload, _address = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                raise ONVIFError(
                    f"Failed while receiving ONVIF discovery responses: {exc}"
                ) from exc
            try:
                devices = parse_discovery_response(payload)
            except ONVIFError:
                continue
            for device in devices:
                key = device.endpoint_reference or device.xaddrs[0]
                discovered.setdefault(key, device)
    finally:
        sock.close()
    return list(discovered.values())


class ONVIFCamera:
    """Authenticated ONVIF Device/Media1 client with GStreamer handoff."""

    def __init__(
        self,
        host: str,
        username: str | None = None,
        password: str | None = None,
        *,
        port: int | None = None,
        timeout: float = 5.0,
        verify_ssl: bool = True,
        time_offset: float = 0.0,
        transport: SOAPTransport | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("ONVIF timeout must be > 0")
        self.device_service_url = self._normalize_device_url(host, port)
        self.username = username
        self._password = password
        self.timeout = float(timeout)
        self.time_offset = float(time_offset)
        self._transport = transport or UrllibSOAPTransport(
            username,
            password,
            verify_ssl=verify_ssl,
        )
        self._media_service_url: str | None = None
        self._profiles: list[ONVIFMediaProfile] | None = None

    @staticmethod
    def _normalize_device_url(host: str, port: int | None) -> str:
        value = host.strip()
        if not value:
            raise ValueError("ONVIF host cannot be empty")
        if "://" not in value:
            value = f"http://{value}"
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid ONVIF host or service URL: '{host}'")
        if port is not None:
            if not 1 <= port <= 65535:
                raise ValueError("ONVIF port must be between 1 and 65535")
            hostname = parsed.hostname
            if ":" in hostname:
                hostname = f"[{hostname}]"
            netloc = f"{hostname}:{port}"
        else:
            netloc = parsed.netloc
        path = parsed.path if parsed.path not in {"", "/"} else "/onvif/device_service"
        return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))

    def _security_header(self, header: ET.Element) -> None:
        if self.username is None:
            return
        nonce = os.urandom(16)
        created = datetime.now(timezone.utc) + timedelta(seconds=self.time_offset)
        created_text = created.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        digest = base64.b64encode(
            hashlib.sha1(nonce + created_text.encode() + (self._password or "").encode()).digest()
        ).decode()
        security = ET.SubElement(
            header,
            f"{{{WSSE_NS}}}Security",
            {f"{{{SOAP_NS}}}mustUnderstand": "true"},
        )
        token = ET.SubElement(security, f"{{{WSSE_NS}}}UsernameToken")
        ET.SubElement(token, f"{{{WSSE_NS}}}Username").text = self.username
        password = ET.SubElement(
            token,
            f"{{{WSSE_NS}}}Password",
            {
                "Type": (
                    "http://docs.oasis-open.org/wss/2004/01/"
                    "oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
                )
            },
        )
        password.text = digest
        nonce_element = ET.SubElement(
            token,
            f"{{{WSSE_NS}}}Nonce",
            {
                "EncodingType": (
                    "http://docs.oasis-open.org/wss/2004/01/"
                    "oasis-200401-wss-soap-message-security-1.0#Base64Binary"
                )
            },
        )
        nonce_element.text = base64.b64encode(nonce).decode()
        ET.SubElement(token, f"{{{WSU_NS}}}Created").text = created_text

    def _call(self, url: str, action: str, operation: ET.Element) -> ET.Element:
        envelope = ET.Element(f"{{{SOAP_NS}}}Envelope")
        header = ET.SubElement(envelope, f"{{{SOAP_NS}}}Header")
        self._security_header(header)
        body = ET.SubElement(envelope, f"{{{SOAP_NS}}}Body")
        body.append(operation)
        payload = ET.tostring(envelope, encoding="utf-8", xml_declaration=True)
        response = self._transport.request(url, action, payload, self.timeout)
        root = _parse_xml(response)
        fault = _first_descendant(root, "Fault")
        if fault is not None:
            raise ONVIFError(_first_text(fault, "Text") or "ONVIF SOAP fault")
        return root

    def _get_services(self) -> list[tuple[str, str]]:
        operation = ET.Element(f"{{{DEVICE_NS}}}GetServices")
        ET.SubElement(operation, f"{{{DEVICE_NS}}}IncludeCapability").text = "false"
        root = self._call(
            self.device_service_url,
            f"{DEVICE_NS}/GetServices",
            operation,
        )
        services: list[tuple[str, str]] = []
        for service in (item for item in root.iter() if _local_name(item.tag) == "Service"):
            namespace = _first_text(service, "Namespace")
            xaddr = _first_text(service, "XAddr")
            if namespace and xaddr:
                services.append((namespace, xaddr))
        return services

    def _get_media_url_from_capabilities(self) -> str | None:
        operation = ET.Element(f"{{{DEVICE_NS}}}GetCapabilities")
        ET.SubElement(operation, f"{{{DEVICE_NS}}}Category").text = "Media"
        root = self._call(
            self.device_service_url,
            f"{DEVICE_NS}/GetCapabilities",
            operation,
        )
        capabilities = _first_descendant(root, "Capabilities")
        if capabilities is None:
            return None
        media = next(
            (item for item in capabilities if _local_name(item.tag) == "Media"),
            None,
        )
        return _first_text(media, "XAddr") if media is not None else None

    def get_media_service_url(self) -> str:
        """Resolve and cache the ONVIF Media1 service endpoint."""
        if self._media_service_url is not None:
            return self._media_service_url
        try:
            services = self._get_services()
        except ONVIFError:
            services = []
        media1 = next((url for namespace, url in services if namespace == MEDIA_NS), None)
        if media1 is None:
            media2 = next(
                (
                    url
                    for namespace, url in services
                    if namespace == "http://www.onvif.org/ver20/media/wsdl"
                ),
                None,
            )
            if media2 is not None:
                raise ONVIFError(
                    "This camera exposes Media2 but not Media1; Media2 is not yet supported"
                )
            media1 = self._get_media_url_from_capabilities()
        if media1 is None:
            raise ONVIFError("Camera did not advertise an ONVIF Media1 service")
        self._media_service_url = media1
        return media1

    def get_profiles(self) -> list[ONVIFMediaProfile]:
        """Return configured Media1 profiles in camera order."""
        if self._profiles is not None:
            return list(self._profiles)
        operation = ET.Element(f"{{{MEDIA_NS}}}GetProfiles")
        root = self._call(
            self.get_media_service_url(),
            f"{MEDIA_NS}/GetProfiles",
            operation,
        )
        profiles: list[ONVIFMediaProfile] = []
        for profile in (item for item in root.iter() if _local_name(item.tag) == "Profiles"):
            token = profile.attrib.get("token")
            if not token:
                continue
            encoder = _first_descendant(profile, "VideoEncoderConfiguration")
            width = _safe_int(_first_text(encoder, "Width")) if encoder is not None else None
            height = _safe_int(_first_text(encoder, "Height")) if encoder is not None else None
            frame_rate = (
                _safe_float(_first_text(encoder, "FrameRateLimit")) if encoder is not None else None
            )
            profiles.append(
                ONVIFMediaProfile(
                    token=token,
                    name=_first_text(profile, "Name") or token,
                    encoding=_first_text(encoder, "Encoding") if encoder is not None else None,
                    width=width,
                    height=height,
                    frame_rate=frame_rate,
                )
            )
        self._profiles = profiles
        return list(profiles)

    def select_profile(self, selector: str | None = None) -> ONVIFMediaProfile:
        """Select the first camera profile, or match a token/name case-insensitively."""
        profiles = self.get_profiles()
        if not profiles:
            raise ONVIFError("Camera returned no media profiles")
        if selector is None:
            return profiles[0]
        exact = next(
            (
                profile
                for profile in profiles
                if profile.token.casefold() == selector.casefold()
                or profile.name.casefold() == selector.casefold()
            ),
            None,
        )
        if exact is None:
            choices = ", ".join(f"{item.name} ({item.token})" for item in profiles)
            raise ValueError(f"unknown ONVIF profile '{selector}'; available: {choices}")
        return exact

    def get_stream_uri(
        self,
        profile: str | ONVIFMediaProfile | None = None,
        *,
        include_credentials: bool = False,
    ) -> str:
        """Resolve the stable RTSP URI for a selected Media1 profile."""
        selected = (
            profile if isinstance(profile, ONVIFMediaProfile) else self.select_profile(profile)
        )
        operation = ET.Element(f"{{{MEDIA_NS}}}GetStreamUri")
        setup = ET.SubElement(operation, f"{{{MEDIA_NS}}}StreamSetup")
        ET.SubElement(setup, f"{{{SCHEMA_NS}}}Stream").text = "RTP-Unicast"
        transport = ET.SubElement(setup, f"{{{SCHEMA_NS}}}Transport")
        ET.SubElement(transport, f"{{{SCHEMA_NS}}}Protocol").text = "RTSP"
        ET.SubElement(operation, f"{{{MEDIA_NS}}}ProfileToken").text = selected.token
        root = self._call(
            self.get_media_service_url(),
            f"{MEDIA_NS}/GetStreamUri",
            operation,
        )
        uri = _first_text(root, "Uri")
        if uri is None or not uri.lower().startswith("rtsp://"):
            raise ONVIFError("Camera returned no valid RTSP stream URI")
        if include_credentials and self.username is not None:
            return _uri_with_credentials(uri, self.username, self._password or "")
        return uri

    def gstreamer_source(
        self,
        profile: str | ONVIFMediaProfile | None = None,
        **kwargs: Any,
    ):
        """Resolve a profile and return a reconnect-capable GStreamerFrameSource."""
        from dfine.utils.sources import GStreamerFrameSource

        uri = self.get_stream_uri(profile)
        kwargs.setdefault("reconnect", True)
        if self.username is not None:
            kwargs.setdefault("rtsp_username", self.username)
            kwargs.setdefault("rtsp_password", self._password or "")
        return GStreamerFrameSource(uri, **kwargs)


def _safe_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _safe_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _uri_with_credentials(uri: str, username: str, password: str) -> str:
    parsed = urllib.parse.urlsplit(uri)
    hostname = parsed.hostname or ""
    if ":" in hostname:
        hostname = f"[{hostname}]"
    if parsed.port is not None:
        hostname += f":{parsed.port}"
    userinfo = urllib.parse.quote(username, safe="")
    if password:
        userinfo += ":" + urllib.parse.quote(password, safe="")
    return urllib.parse.urlunsplit(
        (parsed.scheme, f"{userinfo}@{hostname}", parsed.path, parsed.query, parsed.fragment)
    )
