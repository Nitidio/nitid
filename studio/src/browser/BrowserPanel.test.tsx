import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BackendInfo } from "../api/backend";
import type { LocalApi } from "../api/local";
import type { Model } from "../api/types";
import { BackendContext } from "../backend-context";
import { EngineError } from "../engine/errors";
import { clearToken, initToken } from "../token";
import { engineDetection, fakeEngine, stubImageBitmap } from "../test/engine-fakes";
import { BrowserPanel } from "./BrowserPanel";

const local: BackendInfo = { mode: "local", versionMismatch: false };
const hosted: BackendInfo = { mode: "hosted", versionMismatch: false };
const SHA = "d".repeat(64);

const models: Model[] = [
  { id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" },
  { id: "nitid1m", task: "detect", size: "m", downloaded: false, source: "registry" },
];
const api = { listModels: vi.fn(async () => models) } as unknown as LocalApi;

function renderPanel(backend: BackendInfo | null, props: React.ComponentProps<typeof BrowserPanel> = {}) {
  return render(
    <BackendContext.Provider value={backend}>
      <MemoryRouter>
        <BrowserPanel {...props} />
      </MemoryRouter>
    </BackendContext.Provider>,
  );
}

async function addImage() {
  const input = (await screen.findByLabelText("Images")) as HTMLInputElement;
  fireEvent.change(input, { target: { files: [new File(["x"], "street.jpg", { type: "image/jpeg", lastModified: 1 })] } });
}

beforeEach(() => stubImageBitmap());
afterEach(() => {
  vi.unstubAllGlobals();
  clearToken();
});

describe("BrowserPanel: hosted", () => {
  it("shows a clear empty state and requests nothing when no catalogue is configured", () => {
    const fetchImpl = vi.fn();
    renderPanel(hosted, { hostedBaseUrl: null, fetchImpl: fetchImpl as unknown as typeof fetch });
    expect(screen.getByTestId("no-models")).toHaveTextContent("No models are configured for this site");
    expect(screen.queryByRole("button", { name: "Run" })).toBeNull();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("loads the catalogue, runs the model through the engine and offers exports", async () => {
    const fetchImpl = vi.fn(
      async () =>
        new Response(JSON.stringify({ models: [{ id: "tiny", task: "detect", url: "tiny.onnx", sha256: SHA, label: "Tiny" }] })),
    );
    const engine = fakeEngine();
    renderPanel(hosted, {
      hostedBaseUrl: "https://models.example.test/m/",
      fetchImpl: fetchImpl as unknown as typeof fetch,
      createEngine: () => engine,
    });
    expect(await screen.findByRole("option", { name: "Tiny" })).toBeInTheDocument();
    await addImage();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByRole("heading", { name: "Results" })).toBeInTheDocument();
    expect(engine.init).toHaveBeenCalledWith({ modelUrl: "https://models.example.test/m/tiny.onnx", sha256: SHA }, expect.any(Function));
    expect(engine.run).toHaveBeenCalledWith({ kind: "bitmap", bitmap: expect.anything() }, { conf: 0.1 });
    expect(screen.getByText(/1 of 1 detections shown/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export" })).toBeInTheDocument();
    // Masks outline note is for server results only.
    expect(screen.queryByText(/Mask outlines are not provided/)).toBeNull();
  });

  it("reports a catalogue failure", async () => {
    renderPanel(hosted, {
      hostedBaseUrl: "/m/",
      fetchImpl: (async () => new Response("", { status: 404 })) as typeof fetch,
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 404");
  });
});

describe("BrowserPanel: local deployment", () => {
  it("lists only downloaded server models, fetches the meta with the token and passes it to the engine", async () => {
    initToken({ hash: "#t=tok123", pathname: "/", search: "" } as Location, { state: null, replaceState() {} } as unknown as History);
    const metaCalls: Array<[string, string | null]> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        metaCalls.push([String(url), new Headers(init?.headers).get("authorization")]);
        return new Response(JSON.stringify({ task: "detect", sha256: SHA }));
      }),
    );
    const engine = fakeEngine();
    renderPanel(local, { api, createEngine: () => engine });
    const select = (await screen.findByLabelText("Model and task")) as HTMLSelectElement;
    expect([...select.options].map((o) => o.text)).toEqual(["nitid1s (detect)"]);
    await addImage();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("heading", { name: "Results" });

    expect(metaCalls).toEqual([["/api/v1/onnx/nitid1s/meta?task=detect", "Bearer tok123"]]);
    expect(engine.init).toHaveBeenCalledWith(
      {
        modelUrl: "/api/v1/onnx/nitid1s?task=detect",
        meta: { task: "detect", sha256: SHA },
        sha256: SHA,
        headers: { Authorization: "Bearer tok123" },
      },
      expect.any(Function),
    );
  });

  it("loads the model once and reuses it for a second run", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ task: "detect", sha256: SHA }))));
    const engine = fakeEngine();
    renderPanel(local, { api, createEngine: () => engine });
    await addImage();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    await screen.findByRole("heading", { name: "Results" });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(engine.run).toHaveBeenCalledTimes(2));
    expect(engine.init).toHaveBeenCalledTimes(1);
  });

  it("shows engine errors with a hint and lets the user retry", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ task: "detect" }))));
    const engine = fakeEngine();
    engine.run.mockRejectedValueOnce(new EngineError("image_too_large", "The image has 30 megapixels"));
    renderPanel(local, { api, createEngine: () => engine });
    await addImage();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("30 megapixels");
    expect(alert).toHaveTextContent("20 megapixels");
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it("Stop terminates the worker and does not show an error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ task: "detect" }))));
    const engine = fakeEngine();
    let reject!: (e: unknown) => void;
    engine.run.mockImplementationOnce(() => new Promise((_, r) => (reject = r)));
    renderPanel(local, { api, createEngine: () => engine });
    await addImage();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop" }));
    reject(new EngineError("worker_error", "The inference worker was terminated"));
    await waitFor(() => expect(engine.terminate).toHaveBeenCalled());
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it("draws segmentation masks as outlines", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ task: "segment" }))));
    const mask = { x: 10, y: 10, width: 6, height: 6, data: new Uint8Array(36).fill(1) };
    const engine = fakeEngine("segment", [engineDetection({ has_mask: true, mask })]);
    const segApi = { listModels: vi.fn(async () => [{ id: "nitid1s", task: "segment", size: "s", downloaded: true, source: "registry" }]) } as unknown as LocalApi;
    renderPanel(local, { api: segApi, createEngine: () => engine });
    await addImage();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    await screen.findByRole("heading", { name: "Results" });
    expect(screen.getByRole("checkbox", { name: /Show\s+masks/ })).toBeInTheDocument();
  });
});
