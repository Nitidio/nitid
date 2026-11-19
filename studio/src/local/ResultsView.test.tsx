import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Run } from "../api/types";
import { ResultsView } from "./ResultsView";
import { fromRun } from "./results-model";

const run: Run = {
  id: "r_1",
  model: "nitid1s",
  task: "segment",
  backend: "torch",
  device: "cpu",
  status: "done",
  error_msg: null,
  item_count: 1,
  items: [
    {
      id: 0,
      name: "s.png",
      width: 100,
      height: 50,
      speed: { preprocess: 1, inference: 2, postprocess: 3 },
      detections: [
        {
          box: { x1: 10, y1: 5, x2: 60, y2: 45 },
          confidence: 0.9,
          class_id: 0,
          name: "person",
          has_mask: true,
          segments: { x: [10, 60, 30], y: [5, 5, 45] },
        },
      ],
    },
  ],
};

describe("ResultsView with server outlines", () => {
  it("no longer claims that outlines are missing, and lists the mask", () => {
    render(<ResultsView results={fromRun(run)} loadImage={async () => null} />);
    expect(screen.queryByText(/Mask outlines are not provided/)).toBeNull();
    expect(screen.getByRole("columnheader", { name: "Mask" })).toBeInTheDocument();
    expect(screen.getByText("yes")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Show\s+masks/ })).toBeChecked();
  });
});
