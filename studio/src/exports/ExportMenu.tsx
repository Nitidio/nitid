import { useId, useRef, useState } from "react";
import type { ExportContext, ExportFile } from "./bundle";
import type { BundleFormat, ExportFormat, ExportResults } from "./model";
import { ALL_FORMATS, BUNDLE_FORMATS, IMAGE_FORMATS } from "./model";

export const FORMAT_LABELS: Record<ExportFormat, { label: string; hint: string }> = {
  png: { label: "Annotated image (PNG)", hint: "Boxes, labels and masks drawn on the image" },
  json: { label: "JSON", hint: "All results with the schema documented in the developer guide" },
  csv: { label: "CSV", hint: "One row per detection" },
  yolo: { label: "YOLO TXT", hint: "Normalised labels (polygons for masks) plus classes.txt" },
  crops: { label: "Crops (PNG)", hint: "One image per detection, as a ZIP" },
  pdf: { label: "PDF report", hint: "Metadata, annotated image and detections table" },
  zip: { label: "ZIP bundle", hint: "The checked formats above in one file" },
};

export interface ExportMenuProps {
  results: ExportResults;
  /** Source image (ImageBitmap, canvas or <img>). Without it the image-based formats are disabled. */
  image?: unknown;
  /** Formats offered, in this order. Default: all. */
  formats?: readonly ExportFormat[];
  /** Extra options forwarded to the exporters (raster backend, CSV delimiter, ...). */
  context?: Omit<ExportContext, "image">;
  /** Replaces the browser download (tests, or hosts that save differently). */
  onDownload?: (file: ExportFile) => void;
  onError?: (error: Error) => void;
}

export function saveFile(file: ExportFile): void {
  const url = URL.createObjectURL(file.blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = file.name;
  a.rel = "noopener";
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Download button + format list. Heavy code (ZIP, PDF, canvas work) loads on first use. */
export function ExportMenu({
  results,
  image,
  formats = ALL_FORMATS,
  context,
  onDownload = saveFile,
  onError,
}: ExportMenuProps) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [status, setStatus] = useState("");
  const [selected, setSelected] = useState<ReadonlySet<BundleFormat>>(
    () => new Set(BUNDLE_FORMATS.filter((f) => formats.includes(f))),
  );
  const running = useRef(false);

  const hasImage = image !== undefined && image !== null;
  const empty = results.detections.length === 0;

  const disabledReason = (f: ExportFormat): string | null => {
    if (IMAGE_FORMATS.includes(f) && !hasImage) return "Needs the source image";
    if (f === "crops" && empty) return "No detections to crop";
    return null;
  };

  const run = async (format: ExportFormat): Promise<void> => {
    if (running.current) return;
    running.current = true;
    setBusy(format);
    setStatus(`Preparing ${FORMAT_LABELS[format].label}...`);
    try {
      const bundle = await import("./bundle");
      const ctx: ExportContext = { ...context, image };
      let files: ExportFile[];
      if (format === "zip") {
        const members = BUNDLE_FORMATS.filter((f) => selected.has(f) && !disabledReason(f));
        if (members.length === 0) throw new Error("Select at least one available format for the ZIP");
        files = [await bundle.buildZip(members, results, ctx)];
      } else if (format === "crops") {
        // Several files: always delivered as one ZIP.
        files = [await bundle.buildZip(["crops"], results, ctx)];
      } else {
        files = await bundle.buildFiles(format, results, ctx);
        if (files.length > 1) files = [await bundle.buildZip([format], results, ctx)];
      }
      for (const f of files) onDownload(f);
      setStatus(`${FORMAT_LABELS[format].label} ready.`);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      setStatus(`Export failed: ${err.message}`);
      onError?.(err);
    } finally {
      running.current = false;
      setBusy(null);
    }
  };

  const toggle = (f: BundleFormat): void => {
    const next = new Set(selected);
    if (next.has(f)) next.delete(f);
    else next.add(f);
    setSelected(next);
  };

  const single = formats.filter((f) => f !== "zip");
  const zipOffered = formats.includes("zip");
  return (
    <div className="export-menu">
      <button type="button" aria-expanded={open} aria-controls={`${id}-panel`} onClick={() => setOpen(!open)}>
        Export
      </button>
      {open && (
        <div id={`${id}-panel`} className="export-panel" role="group" aria-label="Export formats">
          <ul className="export-list">
            {single.map((f) => {
              const reason = disabledReason(f);
              const inZip = (BUNDLE_FORMATS as readonly ExportFormat[]).includes(f);
              return (
                <li key={f}>
                  {zipOffered && inZip && (
                    <input
                      type="checkbox"
                      aria-label={`Include ${FORMAT_LABELS[f].label} in the ZIP`}
                      checked={selected.has(f as BundleFormat) && !reason}
                      disabled={reason !== null}
                      onChange={() => toggle(f as BundleFormat)}
                    />
                  )}
                  <span className="export-name">
                    {FORMAT_LABELS[f].label}
                    <small>{reason ?? FORMAT_LABELS[f].hint}</small>
                  </span>
                  <button
                    type="button"
                    disabled={reason !== null || busy !== null}
                    aria-label={`Download ${FORMAT_LABELS[f].label}`}
                    onClick={() => void run(f)}
                  >
                    Download
                  </button>
                </li>
              );
            })}
          </ul>
          {zipOffered && (
            <button type="button" disabled={busy !== null} onClick={() => void run("zip")}>
              Download ZIP of checked formats
            </button>
          )}
        </div>
      )}
      <p role="status" aria-live="polite" className="export-status">
        {status}
      </p>
    </div>
  );
}
