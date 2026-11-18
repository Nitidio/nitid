import { describe, expect, it, vi, type Mock } from "vitest";
import { ApiError, createApiClient } from "./client";

type FetchMock = Mock<typeof fetch>;

function mockFetch(make: () => Response | Promise<Response>): FetchMock {
  return vi.fn<typeof fetch>(async () => make());
}

function initOf(fetchImpl: FetchMock): RequestInit {
  const init = fetchImpl.mock.calls[0]?.[1];
  if (!init) throw new Error("fetch was not called with init");
  return init;
}

function headersOf(fetchImpl: FetchMock): Headers {
  return new Headers(initOf(fetchImpl).headers);
}

describe("createApiClient", () => {
  it("adds the bearer token and the /api/v1 prefix", async () => {
    const fetchImpl = mockFetch(() => new Response('{"a":1}', { status: 200 }));
    const client = createApiClient({ getToken: () => "secret", fetchImpl });
    await expect(client.request<{ a: number }>("/things")).resolves.toEqual({ a: 1 });
    expect(fetchImpl.mock.calls[0]?.[0]).toBe("/api/v1/things");
    expect(headersOf(fetchImpl).get("Authorization")).toBe("Bearer secret");
  });

  it("omits Authorization when there is no token", async () => {
    const fetchImpl = mockFetch(() => new Response("{}", { status: 200 }));
    await createApiClient({ getToken: () => null, fetchImpl }).request("/x");
    expect(headersOf(fetchImpl).has("Authorization")).toBe(false);
  });

  it("sends JSON bodies with a content type", async () => {
    const fetchImpl = mockFetch(() => new Response("{}", { status: 200 }));
    await createApiClient({ getToken: () => null, fetchImpl }).request("/x", {
      method: "POST",
      json: { n: 1 },
    });
    const init = initOf(fetchImpl);
    expect(init.method).toBe("POST");
    expect(init.body).toBe('{"n":1}');
    expect(headersOf(fetchImpl).get("Content-Type")).toBe("application/json");
  });

  it.each([401, 403])("maps %i to an unauthorized error", async (status) => {
    const fetchImpl = mockFetch(() => new Response('{"detail":"bad token"}', { status }));
    const err = await createApiClient({ getToken: () => "t", fetchImpl })
      .request("/x")
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ kind: "unauthorized", status, message: "bad token" });
  });

  it("maps other statuses to an http error with a fallback message", async () => {
    const fetchImpl = mockFetch(() => new Response("boom", { status: 500 }));
    const err = await createApiClient({ getToken: () => null, fetchImpl })
      .request("/x")
      .catch((e: unknown) => e);
    expect(err).toMatchObject({ kind: "http", status: 500, message: "Request failed with status 500" });
  });

  it("maps a 2xx with invalid JSON to invalid-response", async () => {
    const fetchImpl = mockFetch(() => new Response("<html>", { status: 200 }));
    const err = await createApiClient({ getToken: () => null, fetchImpl })
      .request("/x")
      .catch((e: unknown) => e);
    expect(err).toMatchObject({ kind: "invalid-response", status: 200 });
  });

  it("maps a rejected fetch to a network error and keeps the cause", async () => {
    const cause = new TypeError("Failed to fetch");
    const fetchImpl = mockFetch(() => {
      throw cause;
    });
    const err = await createApiClient({ getToken: () => null, fetchImpl })
      .request("/x")
      .catch((e: unknown) => e);
    expect(err).toMatchObject({ kind: "network" });
    expect((err as ApiError).cause).toBe(cause);
  });

  it("maps an aborted request to a timeout error", async () => {
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_res, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("a", "AbortError")));
        }),
    );
    const err = await createApiClient({
      getToken: () => null,
      fetchImpl: fetchImpl as unknown as typeof fetch,
      timeoutMs: 10,
    })
      .request("/x")
      .catch((e: unknown) => e);
    expect(err).toMatchObject({ kind: "timeout" });
  });
});
