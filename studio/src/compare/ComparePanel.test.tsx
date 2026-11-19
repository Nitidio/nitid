import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LocalApi } from "../api/local";
import type { Devices, Model, Run } from "../api/types";
import { engineDetection, fakeEngine, stubImageBitmap } from "../test/engine-fakes";
import { ComparePanel } from "./ComparePanel";

const devices: Devices = { torch: { cpu: true, cuda: [] }, openvino: { available: true, devices: ["CPU"] } };
const models: Model[] = [{ id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" }];
const det = (cls: number, name: string, conf: number, x1: number) => ({
  box: { x1, y1: 10, x2: x1 + 50, y2: 70 }, confidence: conf, class_id: cls, name, has_mask: false,
});
const run: Run = {
  id: "r1", model: "nitid1s", task: "detect", backend: "torch", device: "cpu", status: "done", error_msg: null, item_count: 1,
  items: [{
    id: 0, name: "street.jpg", width: 200, height: 100, speed: { preprocess: 1, inference: 20, postprocess: 1 },
    // person at x=10 (also found by the browser) and a car only the server sees
    detections: [det(0, "person", 0.9, 10), det(2, "car", 0.6, 120)],
  }],
};

function makeApi(): LocalApi {
  return {
    listDevices: vi.fn(async () => devices),
    listModels: vi.fn(async () => models),
    createRun: vi.fn(async () => "r1"),
    getRun: vi.fn(async () => run),
    getItemImage: vi.fn(async () => new Blob(["x"])),
  } as unknown as LocalApi;
}

beforeEach(() => {
  stubImageBitmap();
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ task: "detect", sha256: "e".repeat(64) }))));
});
afterEach(() => vi.unstubAllGlobals());

describe("ComparePanel", () => {
  it("runs server then browser on the same image and summarises the agreement", async () => {
    const api = makeApi();
    const engine = fakeEngine("detect", [engineDetection({ box: { x1: 11, y1: 10, x2: 61, y2: 70 }, confidence: 0.88 })]);
    render(
      <MemoryRouter>
        <ComparePanel api={api} createEngine={() => engine} />
      </MemoryRouter>,
    );
    const input = (await screen.findByLabelText("Images")) as HTMLInputElement;
    expect(screen.getByRole("button", { name: "Compare" })).toBeDisabled();
    fireEvent.change(input, { target: { files: [new File(["x"], "street.jpg", { type: "image/jpeg", lastModified: 1 })] } });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));

    const summary = await screen.findByTestId("agreement", undefined, { timeout: 5000 });
    // person matches (IoU about 0.96), the car exists only on the server: 1 / (2 + 1 - 1) = 50 %.
    expect(summary).toHaveTextContent("Agreement 50%");
    expect(summary).toHaveTextContent("1 matched, 1 only in A, 0 only in B");
    expect(summary).toHaveTextContent("Only in A: car 60%");

    expect(api.createRun).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.createRun).mock.calls[0]![1]).toMatchObject({ backend: "torch", device: "cpu", model: "nitid1s", conf: 0.1 });
    expect(engine.run).toHaveBeenCalledTimes(1);
    expect(engine.terminate).toHaveBeenCalled(); // worker released after the browser side

    expect(screen.getByTestId("A-stats")).toHaveTextContent("2 detections");
    expect(screen.getByTestId("B-stats")).toHaveTextContent("1 detection");
    expect(screen.getByTestId("B-stats")).toHaveTextContent("model load");
    expect(screen.getAllByRole("button", { name: "Export" })).toHaveLength(2);
  });

  it("the shared threshold re-evaluates the agreement", async () => {
    const engine = fakeEngine("detect", [engineDetection({ box: { x1: 10, y1: 10, x2: 60, y2: 70 }, confidence: 0.8 })]);
    render(
      <MemoryRouter>
        <ComparePanel api={makeApi()} createEngine={() => engine} />
      </MemoryRouter>,
    );
    const input = (await screen.findByLabelText("Images")) as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["x"], "s.jpg", { type: "image/jpeg", lastModified: 2 })] } });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    await screen.findByTestId("agreement", undefined, { timeout: 5000 });
    fireEvent.change(screen.getByRole("slider", { name: /Minimum confidence/ }), { target: { value: "70" } });
    // At 70 % the 0.6 car is gone from A: both sides have one matching person.
    await waitFor(() => expect(screen.getByTestId("agreement")).toHaveTextContent("Agreement 100%"));
  });

  it("a failing side is reported and the other side is still shown", async () => {
    const engine = fakeEngine();
    engine.init.mockRejectedValueOnce(new Error("model could not be loaded"));
    render(
      <MemoryRouter>
        <ComparePanel api={makeApi()} createEngine={() => engine} />
      </MemoryRouter>,
    );
    const input = (await screen.findByLabelText("Images")) as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["x"], "s.jpg", { type: "image/jpeg", lastModified: 3 })] } });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    const alert = await screen.findByRole("alert", undefined, { timeout: 5000 });
    expect(within(alert).getByText(/model could not be loaded/)).toBeInTheDocument();
    expect(screen.getByTestId("A-stats")).toBeInTheDocument();
    expect(screen.queryByTestId("agreement")).toBeNull();
  });

  it("explains when there are no downloaded models", async () => {
    const api = { ...makeApi(), listModels: vi.fn(async () => [{ ...models[0]!, downloaded: false }]) } as unknown as LocalApi;
    render(
      <MemoryRouter>
        <ComparePanel api={api} />
      </MemoryRouter>,
    );
    expect(await screen.findByText(/No downloaded models/)).toBeInTheDocument();
  });
});
