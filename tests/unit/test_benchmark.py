from pathlib import Path

import pytest

from dfine import benchmark as benchmark_module


def test_benchmark_exports_requested_formats_and_writes_report(tmp_path, monkeypatch):
    exports = []

    class Model:
        device = "cpu"

        def export(self, **kwargs):
            exports.append(kwargs["format"])
            Path(kwargs["output"]).touch()

    monkeypatch.setattr(
        benchmark_module,
        "_onnx_runner",
        lambda *_args: (lambda: None, lambda: None),
    )
    monkeypatch.setattr(benchmark_module, "_measure", lambda *_args: [2.0, 4.0])

    results, report = benchmark_module.benchmark_model(
        Model(), formats=["onnx"], batch=2, warmup=0, iterations=2, save_dir=tmp_path
    )

    assert exports == ["onnx"]
    assert results[0].latency_ms == 3.0
    assert results[0].fps == pytest.approx(666.666, rel=1e-3)
    assert report.is_file()
    assert '"format": "onnx"' in report.read_text()


def test_benchmark_keeps_other_results_when_runtime_is_unavailable(tmp_path, monkeypatch):
    class Model:
        device = "cpu"

        def export(self, **kwargs):
            Path(kwargs["output"]).touch()

    monkeypatch.setattr(
        benchmark_module,
        "_onnx_runner",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("missing runtime")),
    )

    results, _ = benchmark_module.benchmark_model(
        Model(), formats=["onnx"], warmup=0, iterations=1, save_dir=tmp_path
    )

    assert results[0].latency_ms is None
    assert results[0].status == "skipped: missing runtime"


def test_benchmark_rejects_unknown_format(tmp_path):
    class Model:
        device = "cpu"

    with pytest.raises(ValueError, match="Unsupported benchmark formats"):
        benchmark_module.benchmark_model(Model(), formats=["invalid"], save_dir=tmp_path)
