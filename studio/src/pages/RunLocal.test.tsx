import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import type { BackendInfo } from "../api/backend";
import { BackendContext } from "../backend-context";
import { RunLocal } from "./RunLocal";

function renderWith(backend: BackendInfo | null) {
  return render(
    <BackendContext.Provider value={backend}>
      <MemoryRouter>
        <RunLocal />
      </MemoryRouter>
    </BackendContext.Provider>,
  );
}

describe("RunLocal", () => {
  it("waits while the backend is being detected", () => {
    renderWith(null);
    expect(screen.getByRole("status")).toHaveTextContent("Checking for a local server");
  });

  it("explains how to start the server in hosted mode", () => {
    renderWith({ mode: "hosted", versionMismatch: false });
    expect(screen.getByRole("alert")).toHaveTextContent("nitid studio");
  });

  it("mounts the panel in local mode", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        String(url).endsWith("/devices")
          ? new Response(JSON.stringify({ torch: { cpu: true, cuda: [] }, openvino: { available: false, devices: [] } }))
          : new Response("[]"),
      ),
    );
    renderWith({ mode: "local", versionMismatch: false });
    expect(await screen.findByText("No models are available on this server.")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});
