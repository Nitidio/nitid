import { describe, expect, it } from "vitest";
import { THEME_KEY, applyTheme, effectiveTheme, readStoredTheme, storeTheme } from "./theme";

function fakeStorage(initial: Record<string, string> = {}) {
  const data = { ...initial };
  return {
    getItem: (k: string) => data[k] ?? null,
    setItem: (k: string, v: string) => {
      data[k] = v;
    },
    data,
  };
}

describe("theme", () => {
  it("reads only valid stored values", () => {
    expect(readStoredTheme(fakeStorage({ [THEME_KEY]: "dark" }))).toBe("dark");
    expect(readStoredTheme(fakeStorage({ [THEME_KEY]: "light" }))).toBe("light");
    expect(readStoredTheme(fakeStorage({ [THEME_KEY]: "neon" }))).toBeNull();
    expect(readStoredTheme(fakeStorage())).toBeNull();
    expect(readStoredTheme(null)).toBeNull();
  });

  it("never throws when storage is blocked", () => {
    const broken = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };
    expect(readStoredTheme(broken)).toBeNull();
    expect(() => storeTheme("dark", broken)).not.toThrow();
  });

  it("stores the choice", () => {
    const s = fakeStorage();
    storeTheme("dark", s);
    expect(s.data[THEME_KEY]).toBe("dark");
  });

  it("falls back to the system preference", () => {
    expect(effectiveTheme(null, true)).toBe("dark");
    expect(effectiveTheme(null, false)).toBe("light");
    expect(effectiveTheme("light", true)).toBe("light");
  });

  it("sets and clears data-theme", () => {
    const root = document.createElement("html");
    applyTheme("dark", root);
    expect(root.getAttribute("data-theme")).toBe("dark");
    applyTheme(null, root);
    expect(root.hasAttribute("data-theme")).toBe(false);
  });
});
