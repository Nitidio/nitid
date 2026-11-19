import { describe, expect, it, vi } from "vitest";
import { detectBackend, parseHealth } from "./backend";
import { EXPECTED_API_VERSION } from "./types";

const okBody = { status: "ok", nitid_version: "0.9.6", api_version: EXPECTED_API_VERSION };

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("parseHealth", () => {
  it("keeps an integer api_version as a number", () => {
    expect(parseHealth({ ...okBody, api_version: 1 })?.api_version).toBe(1);
  });

  it("still accepts a legacy string api_version and converts it", () => {
    expect(parseHealth({ ...okBody, api_version: "1" })?.api_version).toBe(1);
    expect(parseHealth({ ...okBody, api_version: " 2 " })?.api_version).toBe(2);
  });

  it.each([
    null,
    "ok",
    {},
    { ...okBody, status: "degraded" },
    { ...okBody, nitid_version: "" },
    { ...okBody, nitid_version: 3 },
    { status: "ok", nitid_version: "1" },
    { ...okBody, api_version: null },
    { ...okBody, api_version: "v1" },
    { ...okBody, api_version: 1.5 },
    { ...okBody, api_version: -1 },
  ])("rejects %j", (value) => {
    expect(parseHealth(value)).toBeNull();
  });
});

describe("detectBackend", () => {
  it("returns local with health on a valid 200", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(okBody));
    const info = await detectBackend({ studioMode: "auto", fetchImpl });
    expect(info).toEqual({ mode: "local", health: okBody, versionMismatch: false });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/v1/health");
    // The liveness probe must not carry the token.
    expect(new Headers(init.headers).has("Authorization")).toBe(false);
  });

  it("flags a version mismatch but stays local", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ ...okBody, api_version: 999 }));
    const info = await detectBackend({ studioMode: "auto", fetchImpl });
    expect(info.mode).toBe("local");
    expect(info.versionMismatch).toBe(true);
  });

  it("treats a legacy string api_version \"1\" as matching", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ ...okBody, api_version: "1" }));
    const info = await detectBackend({ studioMode: "auto", fetchImpl });
    expect(info).toMatchObject({ mode: "local", versionMismatch: false });
    expect(info.health?.api_version).toBe(1);
  });

  it("honours a custom expected api version", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ ...okBody, api_version: 2 }));
    const info = await detectBackend({ studioMode: "auto", fetchImpl, expectedApiVersion: 2 });
    expect(info.versionMismatch).toBe(false);
  });

  it("falls back to hosted on 200 with invalid JSON (SPA fallback HTML)", async () => {
    const fetchImpl = vi.fn(async () => new Response("<!doctype html><html></html>", { status: 200 }));
    expect(await detectBackend({ studioMode: "auto", fetchImpl })).toEqual({
      mode: "hosted",
      versionMismatch: false,
    });
  });

  it("falls back to hosted on 200 with JSON of the wrong shape", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ hello: "world" }));
    expect((await detectBackend({ studioMode: "auto", fetchImpl })).mode).toBe("hosted");
  });

  it("falls back to hosted on 404", async () => {
    const fetchImpl = vi.fn(async () => new Response("nope", { status: 404 }));
    expect((await detectBackend({ studioMode: "auto", fetchImpl })).mode).toBe("hosted");
  });

  it("falls back to hosted when the network is down, without console output", async () => {
    const spies = (["error", "warn", "log"] as const).map((m) =>
      vi.spyOn(console, m).mockImplementation(() => {}),
    );
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    expect((await detectBackend({ studioMode: "auto", fetchImpl })).mode).toBe("hosted");
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  });

  it("falls back to hosted on timeout", async () => {
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    );
    const info = await detectBackend({
      studioMode: "auto",
      fetchImpl: fetchImpl as unknown as typeof fetch,
      timeoutMs: 20,
    });
    expect(info.mode).toBe("hosted");
  });

  it("does not probe at all in a hosted build", async () => {
    const fetchImpl = vi.fn();
    const info = await detectBackend({ studioMode: "hosted", fetchImpl });
    expect(info).toEqual({ mode: "hosted", versionMismatch: false });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("defaults to auto mode in tests (build-time define)", () => {
    expect(__STUDIO_MODE__).toBe("auto");
  });
});
