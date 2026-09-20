import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import axios from "axios";
import { convertDataset } from "../api/datasets";

export function DatasetConverterPage() {
  const [archive, setArchive] = useState<File | null>(null);
  const [target, setTarget] = useState<"coco" | "yolo">("coco");
  const [config, setConfig] = useState("");
  const { mutate, isPending, error } = useMutation({
    mutationFn: () => convertDataset(archive!, target, config.trim() || undefined),
    onSuccess: (blob) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `dataset-${target}.zip`;
      link.click();
      URL.revokeObjectURL(url);
    },
  });

  const message = axios.isAxiosError(error)
    ? error.message
    : error instanceof Error
      ? error.message
      : undefined;

  return (
    <div className="page converter-page">
      <h2>Dataset Converter</h2>
      <p>Convert an archived detection dataset between YOLO text and COCO JSON.</p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (archive) mutate();
        }}
      >
        <label>
          Dataset ZIP
          <input
            type="file"
            accept=".zip,application/zip"
            required
            onChange={(event) => setArchive(event.target.files?.[0] ?? null)}
          />
        </label>
        <label>
          Convert to
          <select value={target} onChange={(event) => setTarget(event.target.value as "coco" | "yolo")}>
            <option value="coco">COCO JSON</option>
            <option value="yolo">YOLO text</option>
          </select>
        </label>
        <label>
          YAML path inside ZIP <span>(optional when the archive has only one)</span>
          <input
            value={config}
            placeholder="data.yaml"
            onChange={(event) => setConfig(event.target.value)}
          />
        </label>
        {message && <p className="error-msg">{message}</p>}
        <button className="btn-primary" type="submit" disabled={!archive || isPending}>
          {isPending ? "Converting..." : "Convert and download"}
        </button>
      </form>
    </div>
  );
}
