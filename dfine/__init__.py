"""
dfine-wrap: Ultralytics-style wrapper for D-FINE object detection.

Usage:
    from dfine import DFINE

    model = DFINE("dfine_l.pth")
    results = model("image.jpg")
    model.train(data="coco.yaml", epochs=50)
    model.export(format="onnx")
"""
from dfine.model import DFINE

__version__ = "0.1.0"
__all__ = ["DFINE"]
