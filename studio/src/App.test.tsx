import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App } from "./App";

describe("App", () => {
  it("detects a local server and enables the Local card", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ status: "ok", nitid_version: "0.9.6", api_version: 1 }), {
          status: 200,
        }),
      ),
    );
    render(<App />);
    expect(await screen.findByText("Connected to a local server.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use local mode" })).toBeEnabled();
    vi.unstubAllGlobals();
  });

  it("falls back to hosted when the backend is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    render(<App />);
    expect(await screen.findByText(/Only in-browser mode is available/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use local mode" })).toBeDisabled();
    vi.unstubAllGlobals();
  });
});
