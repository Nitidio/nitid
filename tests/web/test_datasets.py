import io
import zipfile
from pathlib import Path

import yaml


def _zip_dataset(data_yaml: str) -> bytes:
    root = Path(data_yaml).parent
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(root)
                if relative == Path("data.yml"):
                    config = yaml.safe_load(path.read_text())
                    config["path"] = "."
                    archive.writestr(str(relative), yaml.safe_dump(config))
                else:
                    archive.write(path, relative)
    return stream.getvalue()


def test_dataset_conversion_requires_authentication(client):
    response = client.post(
        "/datasets/convert",
        data={"target": "coco"},
        files={"archive": ("dataset.zip", b"not-a-zip", "application/zip")},
    )
    assert response.status_code == 401


def test_dataset_conversion_returns_download(client, auth_headers, tiny_yolo_dataset):
    response = client.post(
        "/datasets/convert",
        headers=auth_headers,
        data={"target": "coco", "config": "data.yml"},
        files={
            "archive": (
                "dataset.zip",
                _zip_dataset(tiny_yolo_dataset),
                "application/zip",
            )
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
    assert "annotations/instances_train.json" in names
    assert any(name.startswith("configs/datasets/") for name in names)
