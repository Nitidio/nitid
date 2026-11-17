# Interactive demo (source)

`demo/` is the source of the interactive demo ([#7](https://github.com/Nitidio/nitid/issues/7)).
It is a small Gradio app that runs the **published** `nitid` package from PyPI on CPU, so what
visitors try is what `pip install nitid` gives them.

**Status: not published yet** ([#7](https://github.com/Nitidio/nitid/issues/7) stays open until it is).
The folder is laid out as a Hugging Face Space, but where the demo will run has not been decided.

| File | Purpose |
|---|---|
| `demo/app.py` | The app. Plain functions plus `build_demo()`; the only `gradio` import is inside it. |
| `demo/requirements.txt` | Installs `nitid` (and CPU PyTorch wheels). Gradio comes from `sdk_version`. |
| `demo/README.md` | The Space card; the YAML header configures the Space. |
| `demo/examples/` | Example images offered in the interface. |

## Run it locally

```bash
uv sync --extra dev
uv pip install gradio          # not a dependency of nitid
uv run python spaces/demo/app.py
```

## Publish it

Hugging Face's documentation says that creating a Space that runs compute (Gradio or Docker)
requires a paid plan, while static Spaces are free
(<https://huggingface.co/docs/hub/en/spaces-overview>, checked 2026-11-17). So the steps below need
an account in the organization **with a paid plan**. Any other host that runs a container with Python
3.12 and enough RAM for CPU PyTorch (not measured) can also run `demo/app.py`; the app only needs `nitid` and `gradio`.

1. Create the Space: **New Space**, SDK **Gradio**, hardware **CPU basic** (no hourly cost, but see above),
   visibility **Public** (for example `<org>/nitid-demo`).
2. Upload the folder contents to the root of the Space:

   ```bash
   pip install -U huggingface_hub
   hf auth login
   hf upload <org>/nitid-demo spaces/demo . --repo-type=space
   ```

3. Watch the build log on the Space page. The first start installs PyTorch and nitid, and the
   first prediction downloads the weights, so allow several minutes.
4. When it works, link it from the README and nitid.io, and close issue #7.

To update the Space, run the `hf upload` command again after changing `spaces/demo/`.
It is not synchronised automatically.

## Design notes

- `nitid>=0.9.4` with a fallback: releases up to 0.9.4 only know the `model1<size>` names, so
  `load_model` tries `nitid1<size>` first and `model1<size>` second. Once a release with the new
  names is on PyPI, the requirement can be raised and the fallback removed.
- There are no official nano **detection** weights, so size `n` is only offered for segmentation.
- At most two models stay in memory (`lru_cache(maxsize=2)`); a CPU Space has little RAM.
- Tests: `tests/unit/test_space_demo.py` (logic, fake model) and
  `tests/integration/test_space_demo.py` (real tiny models). Neither needs Gradio or network access.
