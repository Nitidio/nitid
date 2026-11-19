import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import type { Devices, Model, Run } from "../api/types";
import { buildBackendOptions, defaultDeviceChoice, isChoiceAvailable } from "./devices";
import { describeError } from "./errors";
import { checkFiles, formatBytes, MAX_IMAGE_BYTES } from "./files";
import { defaultModelKey, groupByTask, modelKey, modelLabel } from "./models";
import { PollTimeoutError, RemoteFailure } from "./polling";
import { fromRun, itemTotalMs } from "./results-model";

const full: Devices = {
  torch: { cpu: true, cuda: [{ index: 0, name: "RTX 4070" }] },
  openvino: { available: true, devices: ["CPU", "GPU", "NPU"] },
};
const bare: Devices = { torch: { cpu: true, cuda: [] }, openvino: { available: false, devices: [] } };

describe("buildBackendOptions", () => {
  it("enables every reported device", () => {
    const [torch, ov] = buildBackendOptions(full);
    expect(torch?.devices.map((d) => [d.value, d.available])).toEqual([
      ["cpu", true],
      ["cuda:0", true],
    ]);
    expect(ov?.devices.map((d) => [d.value, d.available])).toEqual([
      ["CPU", true],
      ["GPU", true],
      ["NPU", true],
    ]);
    expect(torch?.devices[1]?.label).toBe("CUDA 0: RTX 4070");
  });

  it("lists missing hardware as disabled only", () => {
    const [torch, ov] = buildBackendOptions(bare);
    expect(torch?.available).toBe(true);
    expect(torch?.devices.filter((d) => d.available).map((d) => d.value)).toEqual(["cpu"]);
    expect(ov?.available).toBe(false);
    expect(ov?.devices.every((d) => !d.available)).toBe(true);
  });

  it("passes through OpenVINO names such as GPU.1 and marks the family present", () => {
    const [, ov] = buildBackendOptions({ ...bare, openvino: { available: true, devices: ["CPU", "GPU.1"] } });
    expect(ov?.devices.find((d) => d.value === "GPU.1")?.available).toBe(true);
    expect(ov?.devices.find((d) => d.value === "GPU")).toBeUndefined();
    expect(ov?.devices.find((d) => d.value === "NPU")?.available).toBe(false);
  });

  it("chooses torch cpu by default and validates choices", () => {
    const opts = buildBackendOptions(full);
    expect(defaultDeviceChoice(opts)).toEqual({ backend: "torch", device: "cpu" });
    expect(isChoiceAvailable(opts, { backend: "openvino", device: "NPU" })).toBe(true);
    expect(isChoiceAvailable(buildBackendOptions(bare), { backend: "openvino", device: "NPU" })).toBe(false);
  });

  it("falls back to openvino when torch cpu is missing", () => {
    const opts = buildBackendOptions({ torch: { cpu: false, cuda: [] }, openvino: { available: true, devices: ["CPU"] } });
    expect(defaultDeviceChoice(opts)).toEqual({ backend: "openvino", device: "CPU" });
    expect(defaultDeviceChoice(buildBackendOptions({ ...bare, torch: { cpu: false, cuda: [] } }))).toBeNull();
  });
});

describe("models", () => {
  const ms: Model[] = [
    { id: "nitid1s", task: "segment", size: "s", downloaded: false, source: "registry" },
    { id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" },
    { id: "mine", task: "detect", size: null, downloaded: true, source: "local", path: "/m.onnx" },
  ];
  it("treats (id, task) as the identity", () => {
    expect(modelKey(ms[0]!)).not.toBe(modelKey(ms[1]!));
  });
  it("groups by task, detect first", () => {
    expect(groupByTask(ms).map((g) => [g.task, g.models.length])).toEqual([
      ["detect", 2],
      ["segment", 1],
    ]);
  });
  it("prefers a downloaded detect model", () => {
    expect(defaultModelKey(ms)).toBe("detect:nitid1s");
    expect(defaultModelKey([ms[0]!])).toBe("segment:nitid1s");
    expect(defaultModelKey([])).toBeNull();
  });
  it("labels size and local source", () => {
    expect(modelLabel(ms[1]!)).toBe("nitid1s (S)");
    expect(modelLabel(ms[2]!)).toBe("mine [local file]");
  });
});

describe("checkFiles", () => {
  const f = (name: string, type: string, size = 10, lastModified = 1) => {
    const file = new File(["x".repeat(Math.min(size, 10))], name, { type, lastModified });
    Object.defineProperty(file, "size", { value: size });
    return file;
  };
  it("accepts images and rejects other types, empty, oversize and duplicates", () => {
    const a = f("a.jpg", "image/jpeg");
    const { accepted, rejected } = checkFiles(
      [a, f("b.txt", "text/plain"), f("c.png", "image/png", MAX_IMAGE_BYTES + 1), f("d.png", "image/png", 0), f("a.jpg", "image/jpeg")],
      [],
    );
    expect(accepted.map((x) => x.name)).toEqual(["a.jpg"]);
    expect(rejected.map((r) => r.reason)).toEqual(["not an image", "larger than 25 MB", "empty file", "already added"]);
  });
  it("rejects files already queued", () => {
    const a = f("a.jpg", "image/jpeg");
    expect(checkFiles([f("a.jpg", "image/jpeg")], [a]).rejected).toHaveLength(1);
  });
  it("formats sizes", () => {
    expect(formatBytes(500)).toBe("500 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("describeError", () => {
  it("adds a hint for known codes and keeps the code", () => {
    const d = describeError(new ApiError("http", "Too big", 413, { code: "payload_too_large" }));
    expect(d).toMatchObject({ message: "Too big", code: "payload_too_large" });
    expect(d.hint).toMatch(/25 MB/);
  });
  it("treats an unauthorized error without code as unauthorized", () => {
    expect(describeError(new ApiError("unauthorized", "no", 401)).hint).toMatch(/nitid studio/);
  });
  it("explains network failures", () => {
    expect(describeError(new ApiError("network", "x")).hint).toMatch(/still running/);
  });
  it("passes through poll and generic errors", () => {
    expect(describeError(new RemoteFailure("oom")).message).toBe("oom");
    expect(describeError(new PollTimeoutError("The run", 60_000)).message).toMatch(/60 s/);
    expect(describeError("weird").message).toBe("Something went wrong.");
  });
});

describe("fromRun", () => {
  const run: Run = {
    id: "r_1",
    model: "nitid1s",
    task: "detect",
    backend: "openvino",
    device: "CPU",
    status: "done",
    error_msg: null,
    item_count: 1,
    items: [
      {
        id: 0,
        name: "s.jpg",
        width: 10,
        height: 10,
        speed: { preprocess: 1, inference: 2, postprocess: 3 },
        detections: [{ box: { x1: 0, y1: 0, x2: 5, y2: 5 }, confidence: 0.9, class_id: 2, name: "car", has_mask: true }],
      },
    ],
  };
  it("maps snake_case to the UI model", () => {
    const r = fromRun(run, 1234);
    expect(r).toMatchObject({ runId: "r_1", source: "local", backend: "openvino", wallClockMs: 1234 });
    expect(r.items[0]?.detections[0]).toMatchObject({ classId: 2, hasMask: true, name: "car" });
    expect(itemTotalMs(r.items[0]!)).toBe(6);
  });
  it("omits wallClockMs when unknown", () => {
    expect("wallClockMs" in fromRun(run)).toBe(false);
  });
});
