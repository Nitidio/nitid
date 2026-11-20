import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { BackendInfo } from "../api/backend";
import { useBrowserModels } from "./use-browser-models";

const hosted: BackendInfo = { mode: "hosted", versionMismatch: false };
const asFetch = (fn: unknown) => fn as typeof fetch;

describe("useBrowserModels on the hosted site", () => {
  it("is 'unconfigured' and requests nothing without a catalogue base URL", () => {
    const fetchImpl = vi.fn();
    const { result } = renderHook(() => useBrowserModels(hosted, { hostedBaseUrl: null, fetchImpl: asFetch(fetchImpl) }));
    expect(result.current).toEqual({ status: "unconfigured" });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("waits for the backend decision before doing anything", () => {
    const fetchImpl = vi.fn();
    const { result } = renderHook(() => useBrowserModels(null, { hostedBaseUrl: "/models/", fetchImpl: asFetch(fetchImpl) }));
    expect(result.current).toEqual({ status: "loading" });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("loads <base>models.json from the same origin and never touches /api/v1", async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ models: [{ id: "a", task: "detect", url: "a.onnx" }] })));
    const { result } = renderHook(() => useBrowserModels(hosted, { hostedBaseUrl: "/models/", fetchImpl: asFetch(fetchImpl) }));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    const urls = fetchImpl.mock.calls.map((call) => String((call as unknown[])[0]));
    expect(urls).toEqual(["/models/models.json"]);
    expect(urls.some((u) => u.includes("/api/"))).toBe(false);
    if (result.current.status === "ready") {
      expect(result.current.origin).toBe("hosted");
      expect(new URL(result.current.models[0]!.url).pathname).toBe("/models/a.onnx");
    }
  });

  it.each([
    ["the catalogue is unreachable", async () => { throw new TypeError("offline"); }, /could not be reached/],
    ["the server answers an error", async () => new Response("", { status: 503 }), /HTTP 503/],
    // A static host with a single-page-app fallback answers a missing file with index.html and status 200.
    ["a missing models.json is answered with index.html", async () => new Response("<!doctype html><html>"), /not valid JSON/],
    ["the catalogue has the wrong shape", async () => new Response(JSON.stringify({ items: [] })), /must be \{"models": \[\.\.\.\]\}/],
  ])("reports an error when %s", async (_name, impl, message) => {
    const { result } = renderHook(() => useBrowserModels(hosted, { hostedBaseUrl: "/models/", fetchImpl: asFetch(impl) }));
    await waitFor(() => expect(result.current.status).toBe("error"));
    if (result.current.status === "error") expect(result.current.error.message).toMatch(message);
  });

  it("does not report after unmount (aborted)", async () => {
    let release: (r: Response) => void = () => {};
    const fetchImpl = vi.fn(() => new Promise<Response>((r) => (release = r)));
    const { unmount } = renderHook(() => useBrowserModels(hosted, { hostedBaseUrl: "/models/", fetchImpl: asFetch(fetchImpl) }));
    unmount();
    release(new Response(JSON.stringify({ models: [] })));
    await Promise.resolve();
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});
