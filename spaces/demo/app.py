"""Gradio demo for nitid, meant to run as a Hugging Face Space.

Upload an image, pick a task and a model size, and see the detections or the
instance masks drawn on it. Everything runs on CPU, so the default size is the
small one and large uploads are scaled down first.

The logic lives in plain functions (``run_inference``, ``load_model``,
``sizes_for_task``) so it can be unit-tested without Gradio; ``build_demo`` is
the only place that imports ``gradio``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("detect", "segment")
ALL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_TASK = "detect"
DEFAULT_SIZE = "s"
DEFAULT_CONF = 0.5
# Uploads larger than this (longest side, in pixels) are scaled down so a CPU
# Space answers in a reasonable time. The model resizes to 640 anyway.
MAX_SIDE = 1280
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

ModelLoader = Callable[[str, str], Any]


def sizes_for_task(task: str, current: str | None = None) -> tuple[list[str], str]:
    """Return the selectable sizes for ``task`` and the size to keep selected.

    There are no official nano detection weights, so ``n`` is only offered for
    segmentation. ``current`` is kept when it is still valid.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    sizes = [size for size in ALL_SIZES if not (task == "detect" and size == "n")]
    value = current if current in sizes else DEFAULT_SIZE
    return sizes, value


@functools.lru_cache(maxsize=2)
def load_model(task: str, size: str) -> Any:
    """Load (and cache) the ``nitid1<size>`` model for ``task`` on CPU.

    Releases up to 0.9.4 only know the older ``model1<size>`` spelling, so that
    name is tried when ``nitid1<size>`` is not recognised. This keeps the Space
    working with whichever nitid version PyPI serves.
    """
    from nitid import NITID

    last_error: ValueError | None = None
    for name in (f"nitid1{size}", f"model1{size}"):
        try:
            return NITID(name, task=task, device="cpu", verbose=False)
        except ValueError as error:
            if "Unsupported NITID model" not in str(error):
                raise
            last_error = error
    assert last_error is not None
    raise last_error


def _limit_size(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def run_inference(
    image: np.ndarray | None,
    task: str,
    size: str,
    conf: float,
    loader: ModelLoader = load_model,
) -> tuple[np.ndarray, Any, str]:
    """Run one prediction.

    Args:
        image: RGB image as delivered by Gradio.
        task: ``"detect"`` or ``"segment"``.
        size: model size letter, one of ``n s m l x``.
        conf: confidence threshold in ``[0, 1]``.
        loader: returns the model for ``(task, size)``; replaced in tests.

    Returns:
        The annotated RGB image, a pandas DataFrame with one row per object, and a
        one-line summary.
    """
    if image is None:
        raise ValueError("Upload an image first.")
    if size not in sizes_for_task(task)[0]:
        raise ValueError(f"No {task} model is available for size {size!r}.")

    image = _limit_size(np.asarray(image))
    # nitid follows the OpenCV convention (BGR); Gradio hands over RGB.
    bgr = np.ascontiguousarray(image[:, :, ::-1])

    model = loader(task, size)
    result = model.predict(bgr, conf=float(conf), verbose=False)[0]

    annotated = np.ascontiguousarray(result.plot()[:, :, ::-1])
    table = result.pandas()
    if len(table):  # an empty table has object dtype columns, which cannot be rounded
        table = table.round({"x1": 1, "y1": 1, "x2": 1, "y2": 1, "confidence": 3})

    count = len(result)
    summary = f"{count} object{'' if count == 1 else 's'} found"
    inference_ms = result.speed.get("inference")
    if inference_ms:
        summary += f" · model time {inference_ms:.0f} ms on CPU"
    return annotated, table, summary


def build_demo() -> Any:
    """Build the Gradio interface (imports ``gradio`` lazily)."""
    import gradio as gr

    _, initial_size = sizes_for_task(DEFAULT_TASK)
    examples = sorted(str(path) for path in EXAMPLES_DIR.glob("*.jpg"))

    def predict(
        image: np.ndarray | None, task: str, size: str, conf: float
    ) -> tuple[np.ndarray, Any, str]:
        try:
            return run_inference(image, task, size, conf)
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def update_sizes(task: str, size: str) -> Any:
        sizes, value = sizes_for_task(task, size)
        return gr.update(choices=sizes, value=value)

    with gr.Blocks(title="nitid demo") as demo:
        gr.Markdown(
            "# nitid\n"
            "Object detection and instance segmentation with code and weights under "
            "Apache 2.0. Runs here on CPU, so expect a few seconds per image and a "
            "longer first run while the weights download. "
            "[Code](https://github.com/Nitidio/nitid) · "
            "[`pip install nitid`](https://pypi.org/project/nitid/)"
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="Input image")
                task = gr.Radio(list(TASKS), value=DEFAULT_TASK, label="Task")
                # Gradio validates the submitted value against the choices given here,
                # not against later updates, so list every size and narrow the visible
                # ones per task (on load and on change). run_inference rejects the
                # combinations that have no weights.
                size = gr.Dropdown(
                    list(ALL_SIZES),
                    value=initial_size,
                    label="Model size (n smallest, x largest)",
                )
                conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence")
                run = gr.Button("Run", variant="primary")
            with gr.Column():
                output = gr.Image(label="Result")
                summary = gr.Markdown()
                table = gr.Dataframe(label="Objects")

        task.change(update_sizes, inputs=[task, size], outputs=size)
        demo.load(update_sizes, inputs=[task, size], outputs=size)
        run.click(predict, inputs=[image, task, size, conf], outputs=[output, table, summary])
        if examples:
            gr.Examples(examples=examples, inputs=image)
    return demo


if __name__ == "__main__":
    build_demo().queue(max_size=8).launch()
