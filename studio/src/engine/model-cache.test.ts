import { describe, expect, it, vi } from "vitest";
import { MODEL_CACHE_NAME, loadModelBytes, sha256FromEtag, sha256Hex, type CacheLike, type CacheStorageLike } from "./model-cache";

const ORIGIN = "https://studio.test";

/** In-memory Cache API double: insertion-ordered, string keys. */
class FakeCache implements CacheLike {
  readonly store = new Map<string, Uint8Array>();
  failPut = false;
  async match(request: string) {
    const bytes = this.store.get(request);
    return bytes ? new Response(bytes.slice()) : undefined;
  }
  async put(request: string, response: Response) {
    if (this.failPut) throw new Error("QuotaExceededError");
    this.store.set(request, new Uint8Array(await response.arrayBuffer()));
  }
  async delete(request: string) {
    return this.store.delete(request);
  }
  async keys() {
    return [...this.store.keys()].map((url) => ({ url }));
  }
}
class FakeStorage implements CacheStorageLike {
  readonly caches = new Map<string, FakeCache>();
  opened: string[] = [];
  async open(name: string) {
    this.opened.push(name);
    if (!this.caches.has(name)) this.caches.set(name, new FakeCache());
    return this.caches.get(name)!;
  }
  get cache() {
    return this.caches.get(MODEL_CACHE_NAME)!;
  }
}

const bytesOf = (text: string) => new TextEncoder().encode(text);
const hashOf = async (text: string) => sha256Hex(bytesOf(text).buffer as ArrayBuffer);

function fetchStub(body: string, init: { etag?: string | null; headStatus?: number; length?: boolean } = {}) {
  const calls: string[] = [];
  const fn = vi.fn(async (_url: RequestInfo | URL, options?: RequestInit) => {
    const method = options?.method ?? "GET";
    calls.push(method);
    const headers: Record<string, string> = {};
    if (init.etag !== null && init.etag !== undefined) headers.etag = init.etag;
    if (init.length !== false) headers["content-length"] = String(bytesOf(body).length);
    if (method === "HEAD") return new Response(null, { status: init.headStatus ?? 200, headers });
    return new Response(bytesOf(body), { status: 200, headers });
  });
  return { fn: fn as unknown as typeof fetch, calls };
}

describe("model-cache: sha256 helpers", () => {
  it("hashes like the standard (known vector)", async () => {
    expect(await hashOf("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });
  it("extracts only strong, quoted, lowercase 64-hex ETags", () => {
    const hex = "a".repeat(64);
    expect(sha256FromEtag(`"${hex}"`)).toBe(hex);
    expect(sha256FromEtag(`W/"${hex}"`)).toBeNull();
    expect(sha256FromEtag(hex)).toBeNull();
    expect(sha256FromEtag(`"${"A".repeat(64)}"`)).toBeNull();
    expect(sha256FromEtag(null)).toBeNull();
    expect(sha256FromEtag('"abc"')).toBeNull();
  });
});

describe("model-cache: request headers", () => {
  it("sends the headers on the HEAD and the GET, and not when none are given", async () => {
    const seen: Array<{ method: string; auth: string | null }> = [];
    const fn = (async (_url: RequestInfo | URL, options?: RequestInit) => {
      seen.push({ method: options?.method ?? "GET", auth: new Headers(options?.headers).get("authorization") });
      return options?.method === "HEAD"
        ? new Response(null, { status: 200, headers: { etag: `"${await hashOf("m")}"` } })
        : new Response(bytesOf("m"), { status: 200 });
    }) as typeof fetch;
    await loadModelBytes({ url: "/m", headers: { Authorization: "Bearer t" }, fetchFn: fn, caches: new FakeStorage(), origin: ORIGIN });
    expect(seen).toEqual([
      { method: "HEAD", auth: "Bearer t" },
      { method: "GET", auth: "Bearer t" },
    ]);
    seen.length = 0;
    await loadModelBytes({ url: "/m", fetchFn: fn, caches: null, origin: ORIGIN });
    expect(seen).toEqual([{ method: "GET", auth: null }]);
  });
});

describe("model-cache: loadModelBytes", () => {
  it("downloads, verifies the known sha256, caches under it, then serves from cache", async () => {
    const hash = await hashOf("model-bytes");
    const storage = new FakeStorage();
    const { fn, calls } = fetchStub("model-bytes");
    const first = await loadModelBytes({ url: "/onnx/m", sha256: hash, fetchFn: fn, caches: storage, origin: ORIGIN });
    expect(first).toMatchObject({ source: "network", cached: true, sha256: hash });
    expect(new TextDecoder().decode(first.bytes)).toBe("model-bytes");
    expect([...storage.cache.store.keys()]).toEqual([`${ORIGIN}/__nitid-onnx__/${hash}`]);

    const second = await loadModelBytes({ url: "/onnx/m", sha256: hash, fetchFn: fn, caches: storage, origin: ORIGIN });
    expect(second).toMatchObject({ source: "cache", cached: false });
    expect(calls).toEqual(["GET"]); // the second load did not touch the network
  });

  it("without a known sha256 it uses a HEAD request and the strong ETag as key", async () => {
    const hash = await hashOf("etag-model");
    const storage = new FakeStorage();
    const { fn, calls } = fetchStub("etag-model", { etag: `"${hash}"` });
    const first = await loadModelBytes({ url: "/onnx/m", fetchFn: fn, caches: storage, origin: ORIGIN });
    expect(first.source).toBe("network");
    expect(calls).toEqual(["HEAD", "GET"]);
    const second = await loadModelBytes({ url: "/onnx/m", fetchFn: fn, caches: storage, origin: ORIGIN });
    expect(second.source).toBe("cache");
    expect(calls).toEqual(["HEAD", "GET", "HEAD"]);
  });

  it("a new model version has a new key: the old entry is not served", async () => {
    const storage = new FakeStorage();
    const oldHash = await hashOf("v1");
    const newHash = await hashOf("v2");
    await loadModelBytes({ url: "/m", sha256: oldHash, fetchFn: fetchStub("v1").fn, caches: storage, origin: ORIGIN });
    const next = await loadModelBytes({ url: "/m", sha256: newHash, fetchFn: fetchStub("v2").fn, caches: storage, origin: ORIGIN });
    expect(next.source).toBe("network");
    expect(storage.cache.store.size).toBe(2);
  });

  it("does not cache when no key can be known (no meta sha256 and no strong ETag)", async () => {
    const storage = new FakeStorage();
    const { fn } = fetchStub("user-file", { etag: 'W/"weak"' });
    const result = await loadModelBytes({ url: "/m", fetchFn: fn, caches: storage, origin: ORIGIN });
    expect(result).toMatchObject({ source: "network", cached: false });
    expect(storage.cache.store.size).toBe(0);
  });

  it("falls back to a plain GET when HEAD is not supported", async () => {
    const { fn } = fetchStub("x", { headStatus: 405 });
    const result = await loadModelBytes({ url: "/m", fetchFn: fn, caches: new FakeStorage(), origin: ORIGIN });
    expect(result.source).toBe("network");
  });

  it("rejects a download whose hash differs from the expected sha256", async () => {
    const wrong = await hashOf("something else");
    const storage = new FakeStorage();
    await expect(
      loadModelBytes({ url: "/m", sha256: wrong, fetchFn: fetchStub("model").fn, caches: storage, origin: ORIGIN }),
    ).rejects.toMatchObject({ code: "model_hash_mismatch" });
    expect(storage.cache.store.size).toBe(0);
  });

  it("detects a corrupt cache entry, deletes it and downloads again", async () => {
    const hash = await hashOf("good");
    const storage = new FakeStorage();
    const cache = await storage.open(MODEL_CACHE_NAME);
    cache.store.set(`${ORIGIN}/__nitid-onnx__/${hash}`, bytesOf("corrupted"));
    const result = await loadModelBytes({ url: "/m", sha256: hash, fetchFn: fetchStub("good").fn, caches: storage, origin: ORIGIN });
    expect(result.source).toBe("network");
    expect(new TextDecoder().decode(result.bytes)).toBe("good");
    expect(new TextDecoder().decode(cache.store.get(`${ORIGIN}/__nitid-onnx__/${hash}`))).toBe("good");
  });

  it("keeps at most maxEntries models, evicting the oldest", async () => {
    const storage = new FakeStorage();
    const hashes: string[] = [];
    for (const text of ["a", "b", "c", "d"]) {
      const hash = await hashOf(text);
      hashes.push(hash);
      await loadModelBytes({ url: "/m", sha256: hash, fetchFn: fetchStub(text).fn, caches: storage, origin: ORIGIN, maxEntries: 2 });
    }
    expect([...storage.cache.store.keys()]).toEqual(hashes.slice(2).map((h) => `${ORIGIN}/__nitid-onnx__/${h}`));
  });

  it("still works when the cache cannot be written (quota)", async () => {
    const hash = await hashOf("q");
    const storage = new FakeStorage();
    (await storage.open(MODEL_CACHE_NAME)).failPut = true;
    const result = await loadModelBytes({ url: "/m", sha256: hash, fetchFn: fetchStub("q").fn, caches: storage, origin: ORIGIN });
    expect(result).toMatchObject({ source: "network", cached: false });
  });

  it("works without any Cache API (caches: null)", async () => {
    const hash = await hashOf("n");
    const result = await loadModelBytes({ url: "/m", sha256: hash, fetchFn: fetchStub("n").fn, caches: null });
    expect(result).toMatchObject({ source: "network", cached: false });
  });

  it("works when opening the cache throws (private mode)", async () => {
    const hash = await hashOf("p");
    const broken: CacheStorageLike = { open: async () => Promise.reject(new Error("SecurityError")) };
    const result = await loadModelBytes({ url: "/m", sha256: hash, fetchFn: fetchStub("p").fn, caches: broken });
    expect(result.source).toBe("network");
  });

  it("refuses a model larger than the limit from Content-Length, before reading the body", async () => {
    const { fn } = fetchStub("0123456789");
    await expect(loadModelBytes({ url: "/m", fetchFn: fn, caches: null, maxBytes: 5 })).rejects.toMatchObject({
      code: "model_too_large",
    });
  });

  it("refuses a model larger than the limit even without Content-Length (streamed count)", async () => {
    const { fn } = fetchStub("0123456789", { length: false });
    await expect(loadModelBytes({ url: "/m", fetchFn: fn, caches: null, maxBytes: 5 })).rejects.toMatchObject({
      code: "model_too_large",
    });
  });

  it("reports HTTP errors and network failures as model_fetch_failed", async () => {
    const notFound = (async () => new Response("nope", { status: 404 })) as unknown as typeof fetch;
    await expect(loadModelBytes({ url: "/m", fetchFn: notFound, caches: null })).rejects.toMatchObject({
      code: "model_fetch_failed",
      message: expect.stringContaining("404"),
    });
    const offline = (async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;
    await expect(loadModelBytes({ url: "/m", fetchFn: offline, caches: null })).rejects.toMatchObject({
      code: "model_fetch_failed",
    });
  });

  it("reports progress while downloading", async () => {
    const onProgress = vi.fn();
    await loadModelBytes({ url: "/m", fetchFn: fetchStub("progress").fn, caches: null, onProgress });
    expect(onProgress).toHaveBeenCalled();
    const last = onProgress.mock.calls.at(-1)!;
    expect(last[0]).toBe(8);
    expect(last[1]).toBe(8);
  });

  it("validates a malformed sha256 option", async () => {
    await expect(loadModelBytes({ url: "/m", sha256: "XYZ", fetchFn: fetchStub("x").fn, caches: null })).rejects.toMatchObject({
      code: "invalid_options",
    });
  });
});
