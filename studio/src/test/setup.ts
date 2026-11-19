import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
  window.sessionStorage.clear();
});

// jsdom has no 2D canvas; components skip drawing when getContext returns null.
// Drawing itself is covered by src/local/draw.test.ts with a recording fake.
HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
