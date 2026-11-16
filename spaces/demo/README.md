---
title: nitid demo
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Object detection and instance segmentation, Apache 2.0
---

# nitid demo

Upload an image, choose **detect** or **segment** and a model size, and see the result.
The Space runs on CPU: the first request downloads the weights, so expect a wait, and
bigger models take longer. Images above 1280 px on the long side are scaled down.

nitid is a library for DETR-style object detection, instance segmentation and semantic
segmentation with Apache 2.0 code and weights. Code, docs and install instructions:
<https://github.com/Nitidio/nitid>. In your own code:

```python
from nitid import NITID

model = NITID("nitid1s", task="detect")
results = model.predict("image.jpg", conf=0.5)
```

This folder is the source of the Space; see [`../README.md`](../README.md) for how it is
published.
