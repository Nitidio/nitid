# ONVIF cameras

ONVIF supplies camera discovery and configuration metadata. The actual video
still travels over RTSP into the GStreamer pipeline:

```text
WS-Discovery → ONVIF Device service → Media profile → RTSP URI → GStreamer
```

nitid implements IPv4 WS-Discovery, ONVIF Device service discovery, Media1
profiles, `GetStreamUri`, HTTP Basic/Digest authentication, and WS-Security
UsernameToken digest authentication without an additional Python dependency.

## Discover cameras

```bash
dfine onvif action=discover timeout=3
```

Discovery sends a WS-Discovery Probe to `239.255.255.250:3702` and deduplicates
responses by endpoint reference. Select a specific local interface when the
machine has multiple network adapters:

```bash
dfine onvif action=discover timeout=5 interface=192.0.2.20
```

Multicast normally stays within one local network segment. Firewalls, VLANs,
Wi-Fi client isolation, VPN routing, and Docker bridge networking can prevent
responses. On Linux, a discovery container commonly needs host networking:

```bash
docker run --rm --network host nitid-gstreamer \
    dfine onvif action=discover timeout=5
```

Discovery is optional. When the camera address is already known, pass the host
or full device-service URL directly.

## Authentication and profiles

Keep passwords out of process arguments:

```bash
export ONVIF_USERNAME=operator
export ONVIF_PASSWORD='camera password'

dfine onvif action=profiles host=192.0.2.10
```

Use another environment variable when desired:

```bash
export LOBBY_CAMERA_PASSWORD='camera password'
dfine onvif action=profiles \
    host=http://192.0.2.10:8080/onvif/device_service \
    username=operator \
    password_env=LOBBY_CAMERA_PASSWORD
```

The output contains the profile token, display name, codec, resolution, and
frame rate when the camera supplies them. Resolve by token or name:

```bash
dfine onvif action=uri host=192.0.2.10 profile=main
dfine onvif action=uri host=192.0.2.10 profile='Main Stream'
```

The printed RTSP URI intentionally has no inserted username or password.

## Feed tracking securely

Use a separate password environment variable with the normal tracking command:

```bash
export CAMERA_RTSP_PASSWORD='camera password'

dfine track \
    model=nitid1s task=detect \
    source=rtsp://192.0.2.10/Streaming/Channels/101 \
    backend=gstreamer \
    rtsp_username=operator \
    rtsp_password_env=CAMERA_RTSP_PASSWORD \
    hardware_profile=vaapi \
    reconnect=true \
    conf=0.5
```

nitid passes these credentials through `rtspsrc` properties. They are not
inserted into the source URL or written to run metadata. Direct
`rtsp_password=` and ONVIF `password=` CLI arguments are rejected.

## Python API

```python
from dfine import NITID, ONVIFCamera, discover_onvif_devices

devices = discover_onvif_devices(timeout=3, interface="192.0.2.20")
device = devices[0]
if device.service_url is None:
    raise RuntimeError("Camera did not advertise a device-service URL")

camera = ONVIFCamera(
    device.service_url,
    username="operator",
    password="secret",
)

for profile in camera.get_profiles():
    print(profile.token, profile.name, profile.resolution, profile.frame_rate)

source = camera.gstreamer_source(
    "Main Stream",
    hardware_profile="vaapi",
    reconnect=True,
)

model = NITID("nitid1s", task="detect")
for result in model.track(source, stream=True, conf=0.5):
    ...
```

`gstreamer_source()` keeps credentials separate from the RTSP URI and defaults
to reconnection. Pass any normal `GStreamerFrameSource` option to it.

## Direct connection

When multicast discovery is unavailable:

```python
camera = ONVIFCamera(
    "192.0.2.10",
    port=8080,
    username="operator",
    password="secret",
    timeout=5,
)
```

A bare host becomes `http://HOST/onvif/device_service`. A complete HTTP or
HTTPS service URL is preserved. HTTPS certificates are verified by default;
use `verify_ssl=False` only for a camera on a trusted network whose certificate
cannot yet be corrected.

## Clock errors

WS-Security digest authentication includes a UTC creation timestamp. A camera
with an incorrect clock may return `NotAuthorized` even when the credentials
are correct. Correct camera NTP first. As a temporary diagnostic, offset the
client timestamp:

```bash
dfine onvif action=profiles host=192.0.2.10 time_offset=-120
```

```python
camera = ONVIFCamera("192.0.2.10", username="operator", password="secret", time_offset=-120)
```

## Current support boundary

- IPv4 WS-Discovery multicast is supported; IPv6 discovery is not yet included.
- ONVIF Media1 is supported. A Media2-only device produces an explicit error.
- Profile discovery and RTSP URI resolution are read-only; nitid does not alter
  camera configuration.
- Vendor-specific discovery, PTZ, events, analytics configuration, and firmware
  operations are outside this pipeline stage.

The implementation follows the official ONVIF Device and Media service WSDLs
and the ONVIF discovery requirements. Cameras vary in conformance, so validate
each supported model and firmware combination before deployment.
