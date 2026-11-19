import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ExportFile } from "./bundle";
import { ExportMenu } from "./ExportMenu";
import { fakeRaster, results } from "./test-fixtures";

const open = (): void => {
  fireEvent.click(screen.getByRole("button", { name: "Export" }));
};

describe("ExportMenu", () => {
  it("is collapsed until opened and lists the formats", () => {
    render(<ExportMenu results={results()} image="IMG" />);
    const toggle = screen.getByRole("button", { name: "Export" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("group", { name: "Export formats" })).toBeNull();
    open();
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    for (const label of ["Annotated image (PNG)", "JSON", "CSV", "YOLO TXT", "Crops (PNG)", "PDF report"]) {
      expect(screen.getByRole("button", { name: `Download ${label}` })).toBeEnabled();
    }
    expect(screen.getByRole("button", { name: /Download ZIP/ })).toBeEnabled();
  });

  it("downloads a JSON file through onDownload", async () => {
    const onDownload = vi.fn<(f: ExportFile) => void>();
    render(<ExportMenu results={results()} image="IMG" onDownload={onDownload} />);
    open();
    fireEvent.click(screen.getByRole("button", { name: "Download JSON" }));
    await waitFor(() => expect(onDownload).toHaveBeenCalledTimes(1));
    const file = onDownload.mock.calls[0]![0];
    expect(file.name).toBe("street_view.json");
    expect(JSON.parse(new TextDecoder().decode(await file.blob.arrayBuffer())).schema).toBe("nitid-studio-results");
    expect(screen.getByRole("status")).toHaveTextContent("JSON ready.");
  });

  it("delivers multi-file formats (YOLO, crops) as a ZIP", async () => {
    const onDownload = vi.fn<(f: ExportFile) => void>();
    render(<ExportMenu results={results()} image="IMG" context={{ raster: fakeRaster().backend }} onDownload={onDownload} />);
    open();
    fireEvent.click(screen.getByRole("button", { name: "Download YOLO TXT" }));
    await waitFor(() => expect(onDownload).toHaveBeenCalledTimes(1));
    expect(onDownload.mock.calls[0]![0].name).toBe("street_view_export.zip");
    fireEvent.click(screen.getByRole("button", { name: "Download Crops (PNG)" }));
    await waitFor(() => expect(onDownload).toHaveBeenCalledTimes(2));
  });

  it("zips only the checked formats", async () => {
    const onDownload = vi.fn<(f: ExportFile) => void>();
    render(<ExportMenu results={results()} image="IMG" context={{ raster: fakeRaster().backend }} onDownload={onDownload} />);
    open();
    for (const label of ["Annotated image (PNG)", "Crops (PNG)", "PDF report", "YOLO TXT", "CSV"]) {
      fireEvent.click(screen.getByRole("checkbox", { name: `Include ${label} in the ZIP` }));
    }
    fireEvent.click(screen.getByRole("button", { name: /Download ZIP/ }));
    await waitFor(() => expect(onDownload).toHaveBeenCalledTimes(1));
    const { unzipSync } = await import("fflate");
    const files = unzipSync(new Uint8Array(await onDownload.mock.calls[0]![0].blob.arrayBuffer()));
    expect(Object.keys(files)).toEqual(["street_view.json"]);
  });

  it("disables image formats without an image and crops without detections", () => {
    render(<ExportMenu results={results({ detections: [] })} />);
    open();
    expect(screen.getByRole("button", { name: "Download Annotated image (PNG)" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Download PDF report" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Download Crops (PNG)" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Download JSON" })).toBeEnabled();
    expect(screen.getAllByText("Needs the source image").length).toBeGreaterThan(0);
  });

  it("offers only the requested formats", () => {
    render(<ExportMenu results={results()} image="IMG" formats={["csv", "json"]} />);
    open();
    expect(screen.queryByRole("button", { name: /Download PDF/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /Download ZIP/ })).toBeNull();
    expect(screen.getAllByRole("button", { name: /^Download / })).toHaveLength(2);
  });

  it("reports failures through the status line and onError", async () => {
    const onError = vi.fn();
    const failing = { ...fakeRaster().backend, toBlob: () => Promise.reject(new Error("canvas tainted")) };
    render(<ExportMenu results={results()} image="IMG" context={{ raster: failing }} onError={onError} onDownload={vi.fn()} />);
    open();
    fireEvent.click(screen.getByRole("button", { name: "Download Annotated image (PNG)" }));
    await waitFor(() => expect(onError).toHaveBeenCalled());
    expect(screen.getByRole("status")).toHaveTextContent("Export failed: canvas tainted");
    expect(screen.getByRole("button", { name: "Download JSON" })).toBeEnabled();
  });
});
