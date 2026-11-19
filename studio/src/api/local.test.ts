import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "./client";
import { createLocalApi } from "./local";

function setup(respond: (url: string, init: RequestInit) => Response) {
  const fetchImpl = vi.fn<typeof fetch>(async (url, init) => respond(String(url), init ?? {}));
  const api = createLocalApi(createApiClient({ getToken: () => "tok", fetchImpl }));
  return { api, fetchImpl };
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

describe("createLocalApi", () => {
  it("lists devices and models", async () => {
    const { api, fetchImpl } = setup((url) =>
      url.endsWith("/devices")
        ? json({ torch: { cpu: true, cuda: [] }, openvino: { available: false, devices: [] } })
        : json([{ id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" }]),
    );
    expect((await api.listDevices()).torch.cpu).toBe(true);
    expect((await api.listModels())[0]?.id).toBe("nitid1s");
    expect(fetchImpl.mock.calls.map((c) => c[0])).toEqual(["/api/v1/devices", "/api/v1/models"]);
  });

  it("starts a download with the task in the query and returns the job id", async () => {
    const { api, fetchImpl } = setup(() => json({ job_id: "j_1" }, 202));
    await expect(api.startDownload("nitid1s", "segment")).resolves.toBe("j_1");
    expect(fetchImpl.mock.calls[0]?.[0]).toBe("/api/v1/models/nitid1s/download?task=segment");
    expect(fetchImpl.mock.calls[0]?.[1]?.method).toBe("POST");
  });

  it("uploads files and params as multipart with params as a plain string field", async () => {
    const { api, fetchImpl } = setup(() => json({ id: "r_1" }, 202));
    const a = new File(["a"], "a.jpg", { type: "image/jpeg" });
    const b = new File(["b"], "b.png", { type: "image/png" });
    const id = await api.createRun([a, b], { model: "nitid1s", task: "detect", backend: "torch", device: "cpu" });
    expect(id).toBe("r_1");
    const body = fetchImpl.mock.calls[0]?.[1]?.body;
    expect(body).toBeInstanceOf(FormData);
    const form = body as FormData;
    expect(form.getAll("files").map((f) => (f as File).name)).toEqual(["a.jpg", "b.png"]);
    const params = form.get("params");
    expect(typeof params).toBe("string");
    expect(JSON.parse(params as string)).toMatchObject({ model: "nitid1s", backend: "torch" });
    // The browser must set the multipart boundary itself.
    expect(new Headers(fetchImpl.mock.calls[0]?.[1]?.headers).has("Content-Type")).toBe(false);
  });

  it("fetches the item image with the bearer token", async () => {
    const { api, fetchImpl } = setup(() => new Response(new Uint8Array([1]), { status: 200 }));
    const blob = await api.getItemImage("r_1", 2);
    expect(blob.size).toBe(1);
    expect(fetchImpl.mock.calls[0]?.[0]).toBe("/api/v1/runs/r_1/items/2/image");
    expect(new Headers(fetchImpl.mock.calls[0]?.[1]?.headers).get("Authorization")).toBe("Bearer tok");
  });

  it("surfaces the contract error envelope", async () => {
    const { api } = setup(() => json({ error: { code: "not_found", message: "Run r_9 does not exist." } }, 404));
    await expect(api.getRun("r_9")).rejects.toMatchObject({ code: "not_found", status: 404 });
  });
});
