"""Run oriented bounding-box prediction and inspect rotated boxes."""

from nitid import NITID

model = NITID("nitid1s", task="obb")
results = model.predict("aerial.jpg", conf=0.25)

for result in results:
    if result.obb is None:
        continue

    # xywhr: center x, center y, width, height, rotation angle.
    print(result.obb.xywhr)

    # xyxyxyxy: four polygon corners, useful for visualization and YOLO-OBB export.
    print(result.obb.xyxyxyxy)
    result.save("aerial_obb.jpg")
