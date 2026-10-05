"""Build the COCO-mini example dataset from the official COCO val2017 annotations.

COCO-mini is a small COCO-format dataset for smoke-testing detection and instance
segmentation training. Only images whose COCO licence allows commercial use and
derivatives are kept: CC BY 2.0, "No known copyright restrictions" and
"United States Government Work". Annotations keep their original COCO geometry.

Download ``annotations_trainval2017.zip`` from https://cocodataset.org, extract
``instances_val2017.json`` and run::

    uv run python -m tools.build_coco_mini \
        --annotations annotations/instances_val2017.json \
        --output datasets/coco-mini \
        --zip coco-mini.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# COCO licence ids that allow commercial use and derivative works.
ALLOWED_LICENSES = {4, 7, 8}

COCO_ANNOTATION_LICENSE = (
    "COCO annotations © COCO Consortium, licensed under CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/)."
)


def select_images(
    coco: dict,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Select train and val images, covering as many categories as possible in train."""
    categories_by_image: dict[int, set[int]] = {}
    for annotation in coco["annotations"]:
        if not annotation.get("iscrowd", 0):
            categories_by_image.setdefault(annotation["image_id"], set()).add(
                annotation["category_id"]
            )

    candidates = [
        image
        for image in sorted(coco["images"], key=lambda image: image["id"])
        if image["license"] in ALLOWED_LICENSES and image["id"] in categories_by_image
    ]
    if len(candidates) < train_size + val_size:
        raise ValueError(f"Only {len(candidates)} eligible images, need {train_size + val_size}")
    random.Random(seed).shuffle(candidates)

    train: list[dict] = []
    covered: set[int] = set()
    for image in candidates:
        if len(train) == train_size:
            break
        if categories_by_image[image["id"]] - covered:
            train.append(image)
            covered |= categories_by_image[image["id"]]
    chosen = {image["id"] for image in train}
    for image in candidates:
        if len(train) == train_size:
            break
        if image["id"] not in chosen:
            train.append(image)
            chosen.add(image["id"])

    val = [image for image in candidates if image["id"] not in chosen][:val_size]
    return sorted(train, key=lambda image: image["id"]), sorted(val, key=lambda image: image["id"])


def subset(coco: dict, images: list[dict], description: str) -> dict:
    """Return a COCO dict restricted to ``images``, keeping every annotation on them."""
    image_ids = {image["id"] for image in images}
    used_licenses = {image["license"] for image in images}
    return {
        "info": {**coco.get("info", {}), "description": description},
        "licenses": [lic for lic in coco["licenses"] if lic["id"] in used_licenses],
        "images": images,
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in image_ids],
        "categories": coco["categories"],
    }


def flickr_page(flickr_url: str) -> str:
    """Return the Flickr photo page for a static Flickr image URL."""
    match = re.search(r"/(\d+)_[0-9a-f]+(?:_[a-z])?\.jpg$", flickr_url)
    if match is None:
        return flickr_url
    return f"https://www.flickr.com/photo.gne?id={match.group(1)}"


def attribution(splits: dict[str, list[dict]], licenses: list[dict]) -> str:
    """Return the ATTRIBUTION.md text listing the source and licence of every image."""
    license_by_id = {lic["id"]: lic for lic in licenses}
    lines = [
        "# COCO-mini attribution",
        "",
        "COCO-mini is a subset of the COCO 2017 validation set (https://cocodataset.org),",
        "built with `tools/build_coco_mini.py` from the nitid repository.",
        "",
        f"- {COCO_ANNOTATION_LICENSE}",
        "- Each image keeps the licence of its original Flickr photo, listed below.",
        '  Only CC BY 2.0, "No known copyright restrictions" and',
        '  "United States Government Work" images are included.',
        "",
        "| Split | File | Source | Licence |",
        "|-------|------|--------|---------|",
    ]
    for split, images in splits.items():
        for image in images:
            lic = license_by_id[image["license"]]
            lines.append(
                f"| {split} | {image['file_name']} | {flickr_page(image['flickr_url'])} "
                f"| [{lic['name']}]({lic['url']}) |"
            )
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", required=True, help="COCO instances_val2017.json")
    parser.add_argument("--output", default="datasets/coco-mini", help="Dataset directory")
    parser.add_argument("--train", type=int, default=128, help="Number of train images")
    parser.add_argument("--val", type=int, default=32, help="Number of val images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zip", help="Also write a zip archive with the dataset at this path")
    args = parser.parse_args()

    coco = json.loads(Path(args.annotations).read_text())
    train, val = select_images(coco, args.train, args.val, args.seed)
    splits = {"train": train, "val": val}

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    for split, images in splits.items():
        image_dir = output / "images" / split
        image_dir.mkdir(parents=True)
        for image in images:
            urlretrieve(image["coco_url"], image_dir / image["file_name"])
        ann_dir = output / "annotations"
        ann_dir.mkdir(exist_ok=True)
        description = f"COCO-mini {split}: subset of COCO 2017 val with redistributable licences"
        (ann_dir / f"instances_{split}.json").write_text(
            json.dumps(subset(coco, images, description))
        )
    (output / "ATTRIBUTION.md").write_text(attribution(splits, coco["licenses"]))

    covered = {
        ann["category_id"] for ann in subset(coco, train, "")["annotations"] if not ann["iscrowd"]
    }
    print(f"{len(train)} train / {len(val)} val images, {len(covered)} categories in train")

    if args.zip:
        archive_path = Path(args.zip)
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    # Fixed timestamps keep the archive, and its sha256, reproducible.
                    info = zipfile.ZipInfo(
                        str(Path(output.name) / path.relative_to(output)),
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
        print(f"{archive_path}  sha256={sha256(archive_path)}")


if __name__ == "__main__":
    main()
