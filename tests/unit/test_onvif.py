"""Protocol-level tests for ONVIF discovery, Media1, and GStreamer handoff."""

from __future__ import annotations

import socket
import xml.etree.ElementTree as ET

import pytest

from dfine.onvif import (
    MEDIA_NS,
    ONVIFCamera,
    ONVIFError,
    ONVIFMediaProfile,
    build_discovery_probe,
    discover_onvif_devices,
    parse_discovery_response,
)

DISCOVERY_RESPONSE = b"""<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:a="http://www.w3.org/2005/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
 <s:Body><d:ProbeMatches><d:ProbeMatch>
  <a:EndpointReference><a:Address>urn:uuid:camera-1</a:Address></a:EndpointReference>
  <d:Types>dn:NetworkVideoTransmitter tds:Device</d:Types>
  <d:Scopes>onvif://www.onvif.org/name/FrontDoor onvif://www.onvif.org/location/Lobby</d:Scopes>
  <d:XAddrs>http://192.0.2.10/onvif/device_service https://192.0.2.10/onvif/device_service</d:XAddrs>
 </d:ProbeMatch></d:ProbeMatches></s:Body>
</s:Envelope>"""

SERVICES_RESPONSE = b"""<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
 <s:Body><tds:GetServicesResponse><tds:Service>
  <tds:Namespace>http://www.onvif.org/ver10/media/wsdl</tds:Namespace>
  <tds:XAddr>http://192.0.2.10/onvif/media_service</tds:XAddr>
 </tds:Service></tds:GetServicesResponse></s:Body>
</s:Envelope>"""

PROFILES_RESPONSE = b"""<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
 xmlns:tt="http://www.onvif.org/ver10/schema">
 <s:Body><trt:GetProfilesResponse>
  <trt:Profiles token="main"><tt:Name>Main Stream</tt:Name>
   <tt:VideoEncoderConfiguration token="enc-main">
    <tt:Encoding>H264</tt:Encoding><tt:Resolution><tt:Width>1920</tt:Width><tt:Height>1080</tt:Height></tt:Resolution>
    <tt:RateControl><tt:FrameRateLimit>25</tt:FrameRateLimit></tt:RateControl>
   </tt:VideoEncoderConfiguration>
  </trt:Profiles>
  <trt:Profiles token="sub"><tt:Name>Sub Stream</tt:Name>
   <tt:VideoEncoderConfiguration><tt:Encoding>H264</tt:Encoding>
    <tt:Resolution><tt:Width>640</tt:Width><tt:Height>360</tt:Height></tt:Resolution>
   </tt:VideoEncoderConfiguration>
  </trt:Profiles>
 </trt:GetProfilesResponse></s:Body>
</s:Envelope>"""

STREAM_URI_RESPONSE = b"""<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
 <s:Body><trt:GetStreamUriResponse><trt:MediaUri>
  <tt:Uri>rtsp://192.0.2.10:554/Streaming/Channels/101?transport=unicast</tt:Uri>
 </trt:MediaUri></trt:GetStreamUriResponse></s:Body>
</s:Envelope>"""

CAPABILITIES_RESPONSE = b"""<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
 <s:Body><tds:GetCapabilitiesResponse><tds:Capabilities><tt:Media>
  <tt:XAddr>http://192.0.2.10/onvif/media_fallback</tt:XAddr>
 </tt:Media></tds:Capabilities></tds:GetCapabilitiesResponse></s:Body>
</s:Envelope>"""


class FakeTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, url, action, payload, timeout):
        self.calls.append((url, action, payload, timeout))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_discovery_probe_and_response_parser():
    probe = ET.fromstring(build_discovery_probe("urn:uuid:test"))
    texts = [element.text for element in probe.iter()]
    assert "urn:uuid:test" in texts
    assert "urn:schemas-xmlsoap-org:ws:2005:04:discovery" in texts
    assert any(element.tag.endswith("Probe") for element in probe.iter())

    devices = parse_discovery_response(DISCOVERY_RESPONSE)
    assert len(devices) == 1
    device = devices[0]
    assert device.endpoint_reference == "urn:uuid:camera-1"
    assert device.service_url == "https://192.0.2.10/onvif/device_service"
    assert len(device.xaddrs) == 2
    assert "tds:Device" in device.types


def test_discovery_sends_multicast_deduplicates_and_closes_socket():
    class FakeSocket:
        def __init__(self):
            self.responses = [DISCOVERY_RESPONSE, DISCOVERY_RESPONSE, socket.timeout()]
            self.options = []
            self.sent = None
            self.closed = False

        def setsockopt(self, *args):
            self.options.append(args)

        def settimeout(self, timeout):
            self.timeout = timeout

        def sendto(self, payload, address):
            self.sent = (payload, address)

        def recvfrom(self, size):
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response, ("192.0.2.10", 3702)

        def close(self):
            self.closed = True

    fake_socket = FakeSocket()
    clock = iter([0.0, 0.0, 0.1, 0.2, 1.1])
    devices = discover_onvif_devices(
        timeout=1,
        interface="192.0.2.20",
        _socket_factory=lambda *args: fake_socket,
        _clock=lambda: next(clock),
    )

    assert len(devices) == 1
    assert fake_socket.sent[1] == ("239.255.255.250", 3702)
    assert fake_socket.closed
    assert any(option[1] == socket.IP_MULTICAST_IF for option in fake_socket.options)


def test_discovery_validates_timeout_interface_and_xml():
    with pytest.raises(ValueError, match="timeout"):
        discover_onvif_devices(0)
    with pytest.raises(ONVIFError, match="Invalid ONVIF XML"):
        parse_discovery_response("not XML")

    class FakeSocket:
        def setsockopt(self, *args):
            pass

        def close(self):
            self.closed = True

    with pytest.raises(ValueError, match="invalid IPv4"):
        discover_onvif_devices(
            1,
            interface="not-an-ip",
            _socket_factory=lambda *args: FakeSocket(),
        )

    class BlockedSocket(FakeSocket):
        def sendto(self, payload, address):
            raise PermissionError("blocked")

    with pytest.raises(ONVIFError, match="Failed to send"):
        discover_onvif_devices(1, _socket_factory=lambda *args: BlockedSocket())


def test_camera_lists_selects_profiles_and_resolves_stream_uri():
    transport = FakeTransport(
        [SERVICES_RESPONSE, PROFILES_RESPONSE, STREAM_URI_RESPONSE, STREAM_URI_RESPONSE]
    )
    camera = ONVIFCamera(
        "192.0.2.10",
        username="operator",
        password="s/ecret",
        transport=transport,
    )

    profiles = camera.get_profiles()
    assert profiles == [
        ONVIFMediaProfile("main", "Main Stream", "H264", 1920, 1080, 25.0),
        ONVIFMediaProfile("sub", "Sub Stream", "H264", 640, 360, None),
    ]
    assert profiles[0].resolution == (1920, 1080)
    assert camera.select_profile("SUB").token == "sub"

    uri = camera.get_stream_uri(profiles[0])
    assert uri == "rtsp://192.0.2.10:554/Streaming/Channels/101?transport=unicast"
    uri_with_credentials = camera.get_stream_uri(profiles[0], include_credentials=True)
    assert uri_with_credentials.startswith("rtsp://operator:s%2Fecret@192.0.2.10:554/")

    uri_call = transport.calls[2]
    assert uri_call[0] == "http://192.0.2.10/onvif/media_service"
    assert uri_call[1] == f"{MEDIA_NS}/GetStreamUri"
    payload = ET.fromstring(uri_call[2])
    assert any(element.text == "RTP-Unicast" for element in payload.iter())
    assert any(element.text == "RTSP" for element in payload.iter())
    assert any(element.text == "main" for element in payload.iter())


def test_camera_uses_ws_security_digest_without_plaintext_password():
    transport = FakeTransport([SERVICES_RESPONSE, PROFILES_RESPONSE])
    camera = ONVIFCamera(
        "http://camera.local/custom/device",
        username="alice",
        password="plain-secret",
        transport=transport,
    )
    camera.get_profiles()

    payload = transport.calls[0][2]
    root = ET.fromstring(payload)
    assert any(element.text == "alice" for element in root.iter())
    assert b"plain-secret" not in payload
    assert any(element.tag.endswith("Nonce") and element.text for element in root.iter())
    assert any(
        element.tag.endswith("Created") and element.text.endswith("Z") for element in root.iter()
    )


def test_camera_falls_back_to_get_capabilities_when_get_services_fails():
    transport = FakeTransport([ONVIFError("unsupported"), CAPABILITIES_RESPONSE])
    camera = ONVIFCamera("camera.local", transport=transport)

    assert camera.get_media_service_url() == "http://192.0.2.10/onvif/media_fallback"


def test_camera_reports_profile_and_media_errors():
    transport = FakeTransport([SERVICES_RESPONSE, PROFILES_RESPONSE])
    camera = ONVIFCamera("camera.local", transport=transport)
    with pytest.raises(ValueError, match="available: Main Stream"):
        camera.select_profile("missing")

    media2 = b"""<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
      <s:Body><GetServicesResponse><Service><Namespace>http://www.onvif.org/ver20/media/wsdl</Namespace>
      <XAddr>http://camera/onvif/media2</XAddr></Service></GetServicesResponse></s:Body></s:Envelope>"""
    camera2 = ONVIFCamera("camera.local", transport=FakeTransport([media2]))
    with pytest.raises(ONVIFError, match="Media2"):
        camera2.get_media_service_url()


def test_camera_normalizes_urls_and_validates_configuration():
    assert ONVIFCamera("camera.local:8080").device_service_url == (
        "http://camera.local:8080/onvif/device_service"
    )
    assert ONVIFCamera("https://camera.local/path", port=8443).device_service_url == (
        "https://camera.local:8443/path"
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        ONVIFCamera(" ")
    with pytest.raises(ValueError, match="port"):
        ONVIFCamera("camera.local", port=0)
    with pytest.raises(ValueError, match="timeout"):
        ONVIFCamera("camera.local", timeout=0)


def test_camera_handoff_keeps_credentials_out_of_uri(monkeypatch):
    observed = {}

    class FakeSource:
        def __init__(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)

    monkeypatch.setattr("dfine.utils.sources.GStreamerFrameSource", FakeSource)
    camera = ONVIFCamera(
        "camera.local",
        username="operator",
        password="secret",
        transport=FakeTransport([STREAM_URI_RESPONSE]),
    )
    camera._media_service_url = "http://192.0.2.10/onvif/media_service"
    profile = ONVIFMediaProfile("main", "Main")

    source = camera.gstreamer_source(profile, hardware_profile="vaapi")

    assert isinstance(source, FakeSource)
    assert "operator" not in observed["source"]
    assert "secret" not in observed["source"]
    assert observed["kwargs"] == {
        "hardware_profile": "vaapi",
        "reconnect": True,
        "rtsp_username": "operator",
        "rtsp_password": "secret",
    }
