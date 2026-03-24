"""Predict on a single image — the simplest possible usage."""
from dfine import DFINE

model = DFINE("dfine_l.pth")
results = model("image.jpg", conf=0.5)

for r in results:
    print(r.to_json())
    r.save("output.jpg")
