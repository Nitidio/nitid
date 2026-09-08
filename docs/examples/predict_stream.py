"""
Predict on a live stream (RTSP or GStreamer numpy frames).
This is Jorge's primary use case — feed cv2/GStreamer frames directly.
"""

import cv2

from dfine import NITID

model = NITID("nitid1s", task="detect")

# Option A: RTSP stream
for result in model("rtsp://camera_ip/stream", stream=True, conf=0.4):
    annotated = result.plot()
    cv2.imshow("nitid", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

# Option B: numpy frame from your GStreamer pipeline
# frame = your_gstreamer_appsink.pull_sample()  # → np.ndarray HWC BGR
# results = model(frame, conf=0.4)
