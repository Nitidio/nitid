import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import type { BackendInfo } from "../api/backend";
import { BackendContext } from "../backend-context";
import { Home } from "./Home";

function renderHome(backend: BackendInfo | null) {
  return render(
    <BackendContext.Provider value={backend}>
      <MemoryRouter>
        <Home />
      </MemoryRouter>
    </BackendContext.Provider>,
  );
}

const local: BackendInfo = {
  mode: "local",
  health: { status: "ok", nitid_version: "0.9.6", api_version: "1" },
  versionMismatch: false,
};
const hosted: BackendInfo = { mode: "hosted", versionMismatch: false };

describe("Home", () => {
  it("enables both cards in local mode", () => {
    renderHome(local);
    expect(screen.getByRole("button", { name: "Use local mode" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Use browser mode" })).toBeEnabled();
    expect(screen.getByText("Connected to a local server.")).toBeInTheDocument();
  });

  it("disables Local and keeps Browser enabled in hosted mode", () => {
    renderHome(hosted);
    expect(screen.getByRole("button", { name: "Use local mode" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Use browser mode" })).toBeEnabled();
    expect(screen.getByText(/Only in-browser mode is available/)).toBeInTheDocument();
  });

  it("disables both while detecting", () => {
    renderHome(null);
    expect(screen.getByRole("button", { name: "Use local mode" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Use browser mode" })).toBeDisabled();
  });

  it("warns about an API version mismatch", () => {
    renderHome({ ...local, versionMismatch: true });
    expect(screen.getByRole("alert")).toHaveTextContent(/different API version/);
  });

  it("does not warn when versions match", () => {
    renderHome(local);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("never calls localStorage while rendering", () => {
    const getter = vi.fn(() => {
      throw new Error("localStorage must not be used");
    });
    Object.defineProperty(window, "localStorage", { configurable: true, get: getter });
    renderHome(local);
    expect(getter).not.toHaveBeenCalled();
  });
});
