import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { applyViewFilter, ResultsExportMenu } from "./ResultsExportMenu";
import { fromRun, type RunResults } from "../local/results-model";
import { defaultFilter } from "../local/filter";
import type { Run } from "../api/types";

const run: Run = {
  id: "r_1", model: "nitid1s", task: "detect", backend: "torch", device: "cpu", status: "done", error_msg: null, item_count: 1,
  items: [{
    id: 0, name: "a.png", width: 200, height: 100, speed: { preprocess: 1, inference: 2, postprocess: 3 },
    detections: [
      { box: { x1: 1, y1: 1, x2: 50, y2: 50 }, confidence: 0.9, class_id: 0, name: "person", has_mask: false },
      { box: { x1: 60, y1: 1, x2: 90, y2: 50 }, confidence: 0.3, class_id: 2, name: "car", has_mask: false },
    ],
  }],
};

describe("ResultsExportMenu", () => {
  it("maps server results through fromServerRun and applies the on-screen filter", () => {
    const results = fromRun(run);
    const exp = results.toExport!(0);
    expect(exp).toMatchObject({ source: "server", backend: "torch", device: "cpu" });
    expect(applyViewFilter(exp, defaultFilter(0.5)).detections.map((d) => d.name)).toEqual(["person"]);
    expect(applyViewFilter(exp, { minConfidence: 0, hiddenClasses: new Set([0]) }).detections.map((d) => d.name)).toEqual(["car"]);
  });

  it("renders the Export button and opens the format list", () => {
    render(<ResultsExportMenu results={fromRun(run)} filter={defaultFilter(0.25)} selectedItemId={0} />);
    const button = screen.getByRole("button", { name: "Export" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(button);
    expect(screen.getByRole("button", { name: "Download JSON" })).toBeEnabled();
    // No image given: the image-based formats are disabled.
    expect(screen.getByRole("button", { name: "Download PDF report" })).toBeDisabled();
  });

  it("renders nothing for a producer without toExport", () => {
    const results: RunResults = { ...fromRun(run) };
    delete results.toExport;
    const { container } = render(<ResultsExportMenu results={results} filter={defaultFilter()} selectedItemId={0} />);
    expect(container).toBeEmptyDOMElement();
  });
});
