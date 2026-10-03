"""Build a small COCO-format dataset from COCO val2017 for smoke-testing training.

Only images published under CC BY 2.0 are kept, and every selected image is
listed with its Flickr source in ``ATTRIBUTION.txt``. Annotations come from the
official COCO 2017 release (CC BY 4.0). No account or API key is required.

Usage:
    uv run python docs/examples/make_coco_subset.py --out ~/nitid-data/coco-mini
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
CC_BY_LICENSE_ID = 4  # "Attribution License", http://creativecommons.org/licenses/by/2.0/


def _download(url: str, destination: Path, *, quiet: bool = False) -> None:
    if destination.exists():
        return
    if not quiet:
        print(f"Downloading {url}")
    partial = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.rename(destination)


def _load_val_annotations(cache_dir: Path) -> dict:
    json_path = cache_dir / "instances_val2017.json"
    if not json_path.exists():
        zip_path = cache_dir / "annotations_trainval2017.zip"
        _download(ANNOTATIONS_URL, zip_path)  # ~250 MB, downloaded once
        with zipfile.ZipFile(zip_path) as archive:
            json_path.write_bytes(archive.read(ANNOTATIONS_MEMBER))
    with json_path.open() as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--train", type=int, default=64, help="Number of training images")
    parser.add_argument("--val", type=int, default=16, help="Number of validation images")
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    cache_dir = out / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    coco = _load_val_annotations(cache_dir)

    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    candidates = sorted(
        (
            image
            for image in coco["images"]
            if image["license"] == CC_BY_LICENSE_ID
            and any(not a["iscrowd"] for a in annotations_by_image.get(image["id"], []))
        ),
        key=lambda image: image["id"],
    )
    needed = args.train + args.val
    if len(candidates) < needed:
        raise SystemExit(f"Only {len(candidates)} CC BY images available, {needed} requested")

    splits = {"train": candidates[: args.train], "val": candidates[args.train : needed]}
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    attribution = [
        "Images: COCO val2017 (https://cocodataset.org), CC BY 2.0, original Flickr sources below.",
        "Annotations: COCO 2017, CC BY 4.0.",
        "",
    ]
    for split, images in splits.items():
        image_dir = out / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(images)} {split} images")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda image: _download(
                        image["coco_url"], image_dir / image["file_name"], quiet=True
                    ),
                    images,
                )
            )
        attribution.extend(f"{split}/{i['file_name']}  {i['flickr_url']}" for i in images)
        image_ids = {image["id"] for image in images}
        subset = {
            "info": coco["info"],
            "licenses": coco["licenses"],
            "images": images,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids],
            "categories": coco["categories"],
        }
        with (out / "annotations" / f"instances_{split}.json").open("w") as stream:
            json.dump(subset, stream)

    categories = sorted(coco["categories"], key=lambda category: category["id"])
    names = "\n".join(f"  {index}: {c['name']}" for index, c in enumerate(categories))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n\nnc: {len(categories)}\n"
        f"names:\n{names}\n"
    )
    (out / "ATTRIBUTION.txt").write_text("\n".join(attribution) + "\n")
    print(f"Wrote {args.train} train / {args.val} val images and {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
