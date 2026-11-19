import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { BackendInfo } from "./api/backend";
import { AppRoutes } from "./App";
import { BackendContext } from "./backend-context";

const local: BackendInfo = { mode: "local", health: { status: "ok", nitid_version: "0.9.6", api_version: 1 }, versionMismatch: false };
const hosted: BackendInfo = { mode: "hosted", versionMismatch: false };

function at(path: string, backend: BackendInfo | null) {
  return render(
    <BackendContext.Provider value={backend}>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </BackendContext.Provider>,
  );
}

const navNames = () =>
  within(screen.getByRole("navigation", { name: "Main" }))
    .getAllByRole("link")
    .map((a) => a.textContent);

afterEach(() => vi.unstubAllGlobals());

describe("shell and navigation", () => {
  it("local deployment: every entry", () => {
    at("/", local);
    expect(navNames()).toEqual(["Home", "Local", "Browser", "Compare", "About"]);
    expect(screen.getByRole("button", { name: "Open comparison" })).toBeInTheDocument();
  });

  it("hosted deployment: no entry that needs a server", () => {
    at("/", hosted);
    expect(navNames()).toEqual(["Home", "Browser", "About"]);
    expect(screen.queryByRole("button", { name: "Open comparison" })).toBeNull();
  });

  it("while detecting only Home and About are offered", () => {
    at("/", null);
    expect(navNames()).toEqual(["Home", "About"]);
  });

  it("has a skip link and a focusable main landmark", () => {
    at("/", hosted);
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
    expect(screen.getByRole("main")).toHaveAttribute("tabindex", "-1");
  });

  it("hosted /run/browser shows the empty state (nothing is configured in tests)", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    at("/run/browser", hosted);
    expect(await screen.findByTestId("no-models")).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("hosted /compare explains that a server is needed", async () => {
    at("/compare", hosted);
    expect(await screen.findByRole("alert")).toHaveTextContent("nitid studio");
  });

  it("unknown paths, including the old /run/:mode placeholders, give a 404 page", () => {
    at("/run/nonsense", hosted);
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
  });

  it("the old placeholder text is gone", () => {
    at("/run/browser", local);
    expect(screen.queryByText(/not available in this build yet/)).toBeNull();
  });
});
