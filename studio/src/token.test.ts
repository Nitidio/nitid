import { beforeEach, describe, expect, it, vi } from "vitest";
import { clearToken, getToken, initToken, parseTokenFragment } from "./token";

function fakeLocation(hash: string, pathname = "/about", search = "?x=1"): Location {
  return { hash, pathname, search } as Location;
}

function fakeHistory(): History {
  return { state: null, replaceState: vi.fn() } as unknown as History;
}

describe("parseTokenFragment", () => {
  it.each([
    ["#t=abc", "abc"],
    ["t=abc", "abc"],
    ["#foo=1&t=abc", "abc"],
    ["#t=a%2Bb", "a+b"],
  ])("extracts from %s", (hash, expected) => {
    expect(parseTokenFragment(hash)).toBe(expected);
  });

  it.each(["", "#", "#t=", "#other=1", "#section"])("returns null for %j", (hash) => {
    expect(parseTokenFragment(hash)).toBeNull();
  });
});

describe("initToken", () => {
  beforeEach(() => {
    clearToken();
  });

  it("stores the token and strips the fragment, keeping path and query", () => {
    const hist = fakeHistory();
    initToken(fakeLocation("#t=abc"), hist);
    expect(getToken()).toBe("abc");
    expect(hist.replaceState).toHaveBeenCalledWith(null, "", "/about?x=1");
  });

  it("does nothing without a fragment", () => {
    const hist = fakeHistory();
    initToken(fakeLocation(""), hist);
    expect(getToken()).toBeNull();
    expect(hist.replaceState).not.toHaveBeenCalled();
  });

  it("leaves unrelated fragments alone", () => {
    const hist = fakeHistory();
    initToken(fakeLocation("#section"), hist);
    expect(getToken()).toBeNull();
    expect(hist.replaceState).not.toHaveBeenCalled();
  });

  it("works against the real window and history", () => {
    window.history.replaceState(null, "", "/#t=real-token");
    initToken();
    expect(getToken()).toBe("real-token");
    expect(window.location.hash).toBe("");
  });

  it("survives a reload within the tab via sessionStorage only", () => {
    initToken(fakeLocation("#t=abc"), fakeHistory());
    // Simulate a reload: a fresh module would have no memory token.
    // clearToken() would also wipe storage, so check storage directly.
    expect(window.sessionStorage.getItem("nitid-studio.token")).toBe("abc");
  });

  it("clearToken wipes memory and session storage", () => {
    initToken(fakeLocation("#t=abc"), fakeHistory());
    clearToken();
    expect(getToken()).toBeNull();
    expect(window.sessionStorage.getItem("nitid-studio.token")).toBeNull();
  });

  it("never touches localStorage", () => {
    const getter = vi.fn(() => {
      throw new Error("localStorage must not be used");
    });
    Object.defineProperty(window, "localStorage", { configurable: true, get: getter });
    initToken(fakeLocation("#t=abc"), fakeHistory());
    clearToken();
    expect(getter).not.toHaveBeenCalled();
  });
});
