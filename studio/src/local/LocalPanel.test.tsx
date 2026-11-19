import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { LocalApi } from "../api/local";
import type { Devices, Job, Model, Run } from "../api/types";
import { LocalPanel } from "./LocalPanel";
import type { ExportMenuProps } from "./results-model";

const devices: Devices = {
  torch: { cpu: true, cuda: [] },
  openvino: { available: true, devices: ["CPU", "GPU"] },
};
const models: Model[] = [
  { id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" },
  { id: "nitid1m", task: "detect", size: "m", downloaded: false, source: "registry" },
  { id: "nitid1s", task: "segment", size: "s", downloaded: true, source: "registry" },
];
const doneRun: Run = {
  id: "r_1",
  model: "nitid1s",
  task: "detect",
  backend: "torch",
  device: "cpu",
  status: "done",
  error_msg: null,
  item_count: 1,
  items: [
    {
      id: 0,
      name: "street.jpg",
      width: 200,
      height: 100,
      speed: { preprocess: 3.1, inference: 28.4, postprocess: 1.2 },
      detections: [
        { box: { x1: 1, y1: 1, x2: 50, y2: 90 }, confidence: 0.91, class_id: 0, name: "person", has_mask: false },
        { box: { x1: 60, y1: 10, x2: 150, y2: 80 }, confidence: 0.4, class_id: 2, name: "car", has_mask: false },
        { box: { x1: 5, y1: 5, x2: 20, y2: 20 }, confidence: 0.15, class_id: 2, name: "car", has_mask: false },
      ],
    },
  ],
};

function makeApi(over: Partial<LocalApi> = {}): LocalApi {
  return {
    listDevices: vi.fn(async () => devices),
    listModels: vi.fn(async () => models),
    startDownload: vi.fn(async () => "j_1"),
    getJob: vi.fn(async (): Promise<Job> => ({ status: "done", progress: 1 })),
    createRun: vi.fn(async () => "r_1"),
    getRun: vi.fn(async () => doneRun),
    getItemImage: vi.fn(async () => new Blob(["x"])),
    ...over,
  };
}

function renderPanel(api: LocalApi, ExportMenu?: React.ComponentType<ExportMenuProps>) {
  return render(
    <MemoryRouter>
      <LocalPanel api={api} {...(ExportMenu ? { ExportMenu } : {})} />
    </MemoryRouter>,
  );
}

const png = () => new File(["x"], "street.jpg", { type: "image/jpeg", lastModified: 1 });

async function addImage() {
  const input = (await screen.findByLabelText("Images")) as HTMLInputElement;
  fireEvent.change(input, { target: { files: [png()] } });
}

describe("LocalPanel", () => {
  it("loads models and devices and defaults to a downloaded detect model on PyTorch CPU", async () => {
    renderPanel(makeApi());
    const model = (await screen.findByLabelText("Model and task")) as HTMLSelectElement;
    expect(model.value).toBe("detect:nitid1s");
    expect(screen.getByLabelText("Backend")).toHaveValue("torch");
    expect(screen.getByLabelText("Device")).toHaveValue("cpu");
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByText("Add at least one image.")).toBeInTheDocument();
  });

  it("enables only real devices", async () => {
    renderPanel(makeApi());
    await screen.findByLabelText("Device");
    const device = screen.getByLabelText("Device");
    const opt = (name: RegExp) => within(device).getByRole("option", { name }) as HTMLOptionElement;
    expect(opt(/^CPU$/).disabled).toBe(false);
    expect(opt(/CUDA GPU \(not detected\)/).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Backend"), { target: { value: "openvino" } });
    expect(screen.getByLabelText("Device")).toHaveValue("CPU");
    expect(within(screen.getByLabelText("Device")).getByRole("option", { name: /NPU \(not detected\)/ })).toBeDisabled();
  });

  it("runs end to end: upload, poll, show results with timing", async () => {
    const api = makeApi({
      getRun: vi
        .fn<LocalApi["getRun"]>()
        .mockResolvedValueOnce({ ...doneRun, status: "running", items: [] })
        .mockResolvedValue(doneRun),
    });
    renderPanel(api);
    await addImage();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByRole("heading", { name: "Results" }, { timeout: 5000 })).toBeInTheDocument();
    expect(api.createRun).toHaveBeenCalledTimes(1);
    const [files, params] = vi.mocked(api.createRun).mock.calls[0]!;
    expect(files.map((f) => f.name)).toEqual(["street.jpg"]);
    expect(params).toMatchObject({ model: "nitid1s", task: "detect", backend: "torch", device: "cpu", conf: 0.1 });
    expect(screen.getByText(/Preprocess 3\.1 ms, inference 28\.4 ms, postprocess 1\.2 ms/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "street.jpg: 2 detections shown" })).toBeInTheDocument();
  }, 15000);

  it("filters by confidence and by class", async () => {
    renderPanel(makeApi());
    await addImage();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("heading", { name: "Results" }, { timeout: 5000 });

    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(3); // header + person + car 0.4

    fireEvent.change(screen.getByRole("slider", { name: /Minimum confidence/ }), { target: { value: "50" } });
    expect(within(table).getAllByRole("row")).toHaveLength(2);
    expect(screen.getByText(/1 of 3 detections shown/)).toBeInTheDocument();

    fireEvent.change(screen.getByRole("slider", { name: /Minimum confidence/ }), { target: { value: "10" } });
    expect(within(table).getAllByRole("row")).toHaveLength(4);

    fireEvent.click(screen.getByRole("checkbox", { name: /^car/ }));
    expect(within(table).getAllByRole("row")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "None" }));
    expect(screen.getByText("Nothing above the current threshold.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "All" }));
    expect(within(table).getAllByRole("row")).toHaveLength(4);
  }, 15000);

  it("renders the ExportMenu slot with results, filter and selected item", async () => {
    const seen: ExportMenuProps[] = [];
    const Menu = (p: ExportMenuProps) => {
      seen.push(p);
      return <button type="button">Export (slot)</button>;
    };
    renderPanel(makeApi(), Menu);
    await addImage();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByRole("button", { name: "Export (slot)" }, { timeout: 5000 })).toBeInTheDocument();
    const last = seen.at(-1)!;
    expect(last.results.runId).toBe("r_1");
    expect(last.selectedItemId).toBe(0);
    expect(last.filter.minConfidence).toBeCloseTo(0.25);
  }, 15000);

  it("shows the contract error message and hint when the upload is rejected", async () => {
    const api = makeApi({
      createRun: vi.fn(async () => {
        throw new ApiError("http", "Image exceeds 25 MB", 413, { code: "payload_too_large" });
      }),
    });
    renderPanel(api);
    await addImage();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const alert = await screen.findByText(/Image exceeds 25 MB/);
    expect(alert.closest("[role=alert]")).toHaveTextContent("payload_too_large");
    expect(alert.closest("[role=alert]")).toHaveTextContent(/25 MB and 20 megapixels/);
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it("shows the run's error_msg when the server reports failed", async () => {
    const api = makeApi({ getRun: vi.fn(async () => ({ ...doneRun, status: "failed" as const, error_msg: "model file corrupt", items: [] })) });
    renderPanel(api);
    await addImage();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByText("model file corrupt", {}, { timeout: 5000 })).toBeInTheDocument();
  }, 15000);

  it("explains an expired token and lets the user retry the initial load", async () => {
    const listModels = vi
      .fn<LocalApi["listModels"]>()
      .mockRejectedValueOnce(new ApiError("unauthorized", "Invalid token", 401, { code: "unauthorized" }))
      .mockResolvedValue(models);
    renderPanel(makeApi({ listModels }));
    expect(await screen.findByText(/Invalid token/)).toBeInTheDocument();
    expect(screen.getByText(/Open Studio from the link printed by `nitid studio`/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByLabelText("Model and task")).toBeInTheDocument();
  });

  it("downloads a missing model with progress and then enables it", async () => {
    let downloaded = false;
    const progress = [0.3, 0.7, 1];
    const api = makeApi({
      listModels: vi.fn(async () => models.map((m) => (m.id === "nitid1m" && downloaded ? { ...m, downloaded: true } : m))),
      getJob: vi.fn(async (): Promise<Job> => {
        const p = progress.shift() ?? 1;
        if (p === 1) downloaded = true;
        return { status: p === 1 ? "done" : "running", progress: p };
      }),
    });
    renderPanel(api);
    const select = await screen.findByLabelText("Model and task");
    fireEvent.change(select, { target: { value: "detect:nitid1m" } });
    expect(screen.getByText("Not downloaded yet.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Download nitid1m" }));
    expect(await screen.findByText(/Downloading nitid1m/)).toBeInTheDocument();
    expect(api.startDownload).toHaveBeenCalledWith("nitid1m", "detect", expect.anything());
    await waitFor(() => expect(screen.getByText("Ready on this computer.")).toBeInTheDocument(), { timeout: 5000 });
  }, 15000);

  it("shows a failed download without losing the picker", async () => {
    const api = makeApi({ getJob: vi.fn(async (): Promise<Job> => ({ status: "failed", progress: 0.1, error: "network unreachable" })) });
    renderPanel(api);
    fireEvent.change(await screen.findByLabelText("Model and task"), { target: { value: "detect:nitid1m" } });
    fireEvent.click(screen.getByRole("button", { name: "Download nitid1m" }));
    expect(await screen.findByText("network unreachable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download nitid1m" })).toBeEnabled();
  });

  it("validates files and allows removing them", async () => {
    renderPanel(makeApi());
    const input = (await screen.findByLabelText("Images")) as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["x"], "notes.txt", { type: "text/plain" }), png()] } });
    expect(screen.getByRole("alert")).toHaveTextContent("notes.txt: not an image");
    expect(screen.getByRole("button", { name: "Remove street.jpg" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Remove street.jpg" }));
    expect(screen.queryByRole("button", { name: "Remove street.jpg" })).not.toBeInTheDocument();
  });

  it("accepts dropped images", async () => {
    renderPanel(makeApi());
    await screen.findByLabelText("Images");
    const zone = screen.getByText("Drop images here, or").parentElement!;
    fireEvent.drop(zone, { dataTransfer: { files: [png()] } });
    expect(screen.getByRole("button", { name: "Remove street.jpg" })).toBeInTheDocument();
  });
});
