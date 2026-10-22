"""Export a nitid model for deployment.

Every exported graph includes the postprocessor, so it returns
``(labels, boxes, scores)``. See docs/export.md for the options of each format.
"""

from nitid import NITID

model = NITID("model1s", task="detect")

# ONNX: any ONNX runtime, CPU or GPU
onnx_path = model.export(format="onnx", imgsz=640, simplify=True, opset=17)

# OpenVINO: Intel CPU, iGPU and NPU (pip install "nitid[openvino]")
# xml_path = model.export(format="openvino", imgsz=640)

# TorchScript: LibTorch (C++)
# ts_path = model.export(format="torchscript", imgsz=640)

# TensorRT: NVIDIA GPU, needs tensorrt>=8.6 and a CUDA device
# engine_path = model.export(format="tensorrt", imgsz=640, batch=1)
