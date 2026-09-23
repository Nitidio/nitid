"""Export to TensorRT for deployment in GStreamer pipeline."""

from nitid import NITID

model = NITID("nitid1s", task="detect")

# Step 1: ONNX
onnx_path = model.export(format="onnx", imgsz=640, simplify=True, opset=17)

# Step 2: TensorRT (Phase 2)
# trt_path = model.export(format="tensorrt", imgsz=640, batch=1)
