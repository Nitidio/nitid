"""Unit tests for the COCO-mini dataset builder."""

from tools.build_coco_mini import ALLOWED_LICENSES, flickr_page, select_images, subset


def _coco():
    images = [{"id": index, "license": 4 if index % 3 else 2} for index in range(1, 31)]
    annotations = [
        {"id": index, "image_id": index, "category_id": index % 5 + 1, "iscrowd": 0}
        for index in range(1, 31)
    ]
    annotations.append({"id": 100, "image_id": 1, "category_id": 3, "iscrowd": 1})
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": f"c{index}"} for index in range(1, 6)],
        "licenses": [{"id": 2, "name": "NC", "url": ""}, {"id": 4, "name": "BY", "url": ""}],
    }


def test_select_images_keeps_only_allowed_licences_and_disjoint_splits():
    train, val = select_images(_coco(), train_size=8, val_size=4, seed=0)

    assert len(train) == 8
    assert len(val) == 4
    assert {image["license"] for image in train + val} <= ALLOWED_LICENSES
    assert not {image["id"] for image in train} & {image["id"] for image in val}


def test_select_images_covers_categories_first():
    coco = _coco()
    train, _ = select_images(coco, train_size=5, val_size=1, seed=0)
    train_ids = {image["id"] for image in train}

    covered = {ann["category_id"] for ann in coco["annotations"] if ann["image_id"] in train_ids}
    assert covered == {1, 2, 3, 4, 5}


def test_select_images_is_deterministic():
    assert select_images(_coco(), 8, 4, seed=3) == select_images(_coco(), 8, 4, seed=3)


def test_subset_keeps_all_annotations_and_used_licences_only():
    coco = _coco()
    images = [image for image in coco["images"] if image["id"] == 1]

    result = subset(coco, images, "mini")

    assert [ann["id"] for ann in result["annotations"]] == [1, 100]
    assert [lic["id"] for lic in result["licenses"]] == [4]
    assert result["categories"] == coco["categories"]
    assert result["info"]["description"] == "mini"


def test_flickr_page_extracts_photo_id():
    url = "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg"

    assert flickr_page(url) == "https://www.flickr.com/photo.gne?id=6255196340"
    assert flickr_page("http://example.test/other.png") == "http://example.test/other.png"
