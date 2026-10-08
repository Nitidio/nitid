"""
Predict on a live stream (an RTSP URL or numpy frames from your own capture code).
"""

import cv2

from nitid import NITID

model = NITID("model1s", task="detect")

# Option A: RTSP stream
for result in model("rtsp://camera_ip/stream", stream=True, conf=0.4):
    annotated = result.plot()
    cv2.imshow("nitid", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

# Option B: numpy frame from your own capture code
# frame = your_capture.read()  # → np.ndarray HWC BGR
# results = model(frame, conf=0.4)
