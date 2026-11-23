import { describe, expect, it, vi } from "vitest";
import type { Model } from "../api/types";
import { CatalogError, fromServerModels, loadHostedCatalog, parseHostedCatalog, serverMetaPath, serverOnnxUrl } from "./catalog";
import { normalizeBaseUrl, validateModelsBaseUrl } from "../config";

const SHA = "a".repeat(64);
const BASE = "https://models.example.test/nitid/";

describe("config", () => {
  it("normalises the base URL and treats empty as not configured", () => {
    expect(normalizeBaseUrl("")).toBeNull();
    expect(normalizeBaseUrl("   ")).toBeNull();
    expect(normalizeBaseUrl(undefined)).toBeNull();
    expect(normalizeBaseUrl("/models")).toBe("/models/");
    expect(normalizeBaseUrl("https://x.test/m/")).toBe("https://x.test/m/");
  });
});

describe("server catalogue", () => {
  const models: Model[] = [
    { id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" },
    { id: "nitid1s", task: "segment", size: "s", downloaded: false, source: "registry" },
    { id: "my model", task: "detect", size: null, downloaded: true, source: "local", path: "/x" },
  ];
  it("lists only downloaded models, with the ONNX endpoint as URL", () => {
    const list = fromServerModels(models);
    expect(list.map((m) => m.key)).toEqual(["detect:nitid1s", "detect:my model"]);
    expect(list[0]?.url).toBe("/api/v1/onnx/nitid1s?task=detect");
    expect(list[1]?.url).toBe("/api/v1/onnx/my%20model?task=detect");
    expect(list.every((m) => m.origin === "server")).toBe(true);
  });
  it("builds the meta path", () => {
    expect(serverMetaPath("nitid1s", "segment")).toBe("/onnx/nitid1s/meta?task=segment");
    expect(serverOnnxUrl("a/b", "detect")).toBe("/api/v1/onnx/a%2Fb?task=detect");
  });
});

describe("hosted catalogue", () => {
  it("resolves urls against the base and keeps sha256 and label", () => {
    const [m] = parseHostedCatalog(
      { models: [{ id: "nitid1s", task: "detect", url: "nitid1s.onnx", sha256: SHA, label: "Small" }] },
      BASE,
    );
    expect(m).toEqual({
      key: "detect:nitid1s",
      id: "nitid1s",
      task: "detect",
      label: "Small",
      origin: "hosted",
      url: "https://models.example.test/nitid/nitid1s.onnx",
      sha256: SHA,
    });
  });
  it("accepts a root-relative base", () => {
    const [m] = parseHostedCatalog({ models: [{ id: "a", task: "segment", url: "a.onnx" }] }, "/models/");
    expect(m?.url.endsWith("/models/a.onnx")).toBe(true);
    expect(new URL(m!.url).origin).toBe(globalThis.location.origin);
    expect(m?.sha256).toBeUndefined();
  });
  describe("absolute base URL (models on another origin)", () => {
    const ABS = "https://models.example.test/";
    const parse = (url: string) => parseHostedCatalog({ models: [{ id: "a", task: "detect", url }] }, ABS)[0]!.url;
    it("resolves relative urls against it", () => {
      expect(parse("a.onnx")).toBe("https://models.example.test/a.onnx");
      expect(parse("./sub/a.onnx")).toBe("https://models.example.test/sub/a.onnx");
      expect(parse("/a.onnx")).toBe("https://models.example.test/a.onnx");
    });
    it("keeps a base path", () => {
      const [m] = parseHostedCatalog({ models: [{ id: "a", task: "detect", url: "a.onnx" }] }, "https://h.test/m/v1/");
      expect(m?.url).toBe("https://h.test/m/v1/a.onnx");
    });
    it("accepts an absolute url on the same origin", () => {
      expect(parse("https://models.example.test/x/a.onnx")).toBe("https://models.example.test/x/a.onnx");
    });
    it.each(["https://cdn.test/a.onnx", "http://models.example.test/a.onnx", "https://models.example.test:8443/a.onnx", "//cdn.test/a.onnx"])(
      "refuses %s: the CSP allows only the models origin",
      (url) => {
        expect(() => parse(url)).toThrow(CatalogError);
        expect(() => parse(url)).toThrow(/must be on the models origin https:\/\/models\.example\.test/);
      },
    );
    it("refuses a url that cannot be parsed", () => {
      expect(() => parse("http://")).toThrow(/not a valid URL/);
    });
    it("fetches models.json from it with CORS and no credentials", async () => {
      const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ models: [{ id: "a", task: "detect", url: "a.onnx" }] })));
      const [m] = await loadHostedCatalog(ABS, fetchImpl as unknown as typeof fetch);
      expect(m?.url).toBe("https://models.example.test/a.onnx");
      const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
      expect(url).toBe("https://models.example.test/models.json");
      expect(init).toMatchObject({ mode: "cors", credentials: "omit", headers: { Accept: "application/json" } });
    });
  });
  it.each([
    [null, /\{"models"/],
    [{ models: "x" }, /\{"models"/],
    [{ models: [1] }, /not an object/],
    [{ models: [{ task: "detect", url: "a" }] }, /\.id/],
    [{ models: [{ id: "a", task: "pose", url: "a" }] }, /\.task/],
    [{ models: [{ id: "a", task: "detect" }] }, /\.url/],
    [{ models: [{ id: "a", task: "detect", url: "a", sha256: "ABC" }] }, /sha256/],
    [{ models: [{ id: "a", task: "detect", url: "a" }, { id: "a", task: "detect", url: "b" }] }, /duplicate/],
  ])("rejects %j", (body, message) => {
    expect(() => parseHostedCatalog(body, BASE)).toThrow(CatalogError);
    expect(() => parseHostedCatalog(body, BASE)).toThrow(message);
  });
  it("fetches models.json under the base", async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ models: [{ id: "a", task: "detect", url: "a.onnx" }] })));
    const list = await loadHostedCatalog(BASE, fetchImpl as unknown as typeof fetch);
    expect(list).toHaveLength(1);
    expect((fetchImpl.mock.calls[0] as unknown[])[0]).toBe(`${BASE}models.json`);
  });
  it("turns network, HTTP and JSON failures into CatalogError", async () => {
    await expect(loadHostedCatalog(BASE, (async () => { throw new TypeError("x"); }) as typeof fetch)).rejects.toThrow(/could not be reached/);
    await expect(loadHostedCatalog(BASE, (async () => new Response("", { status: 404 })) as typeof fetch)).rejects.toThrow(/HTTP 404/);
    await expect(loadHostedCatalog(BASE, (async () => new Response("<html>")) as typeof fetch)).rejects.toThrow(/not valid JSON/);
  });
});

describe("STUDIO_MODELS_BASE_URL validation (build time)", () => {
  it.each(["", "   ", "/models/", "/models", "https://models.example.test/", "https://models.example.test/a/b/", "http://localhost:8080/", "http://127.0.0.1:9000/m/"])(
    "accepts %j",
    (value) => {
      expect(validateModelsBaseUrl(value)).toBe(value.trim());
    },
  );
  it("treats undefined as not configured", () => {
    expect(validateModelsBaseUrl(undefined)).toBe("");
  });
  it.each([
    ["http://models.example.test/", /https/],
    ["ftp://models.example.test/", /https/],
    ["//models.example.test/", /protocol-relative/],
    ["models.example.test/", /absolute/],
    ["https://user:pw@models.example.test/", /credentials/],
    ["https://models.example.test/?x=1", /query/],
    ["https://models.example.test/#f", /query/],
    ["/models/?x=1", /query/],
  ])("refuses %s", (value, message) => {
    expect(() => validateModelsBaseUrl(value)).toThrow(message);
    expect(() => validateModelsBaseUrl(value)).toThrow(/STUDIO_MODELS_BASE_URL/);
  });
});
