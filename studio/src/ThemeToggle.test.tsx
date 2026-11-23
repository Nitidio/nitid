import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { ThemeToggle } from "./ThemeToggle";
import { Logo } from "./brand/Logo";
import { THEME_KEY } from "./theme";

describe("ThemeToggle", () => {
  beforeEach(() => {
    try {
      sessionStorage.clear();
    } catch {
      /* ignore */
    }
    document.documentElement.removeAttribute("data-theme");
  });

  it("switches the theme, persists it and updates its label", () => {
    render(<ThemeToggle />);
    const button = screen.getByRole("button", { name: /switch to (dark|light) theme/i });
    const before = button.getAttribute("aria-label");
    fireEvent.click(button);
    const stored = sessionStorage.getItem(THEME_KEY);
    expect(["light", "dark"]).toContain(stored);
    expect(document.documentElement.getAttribute("data-theme")).toBe(stored);
    expect(button.getAttribute("aria-label")).not.toBe(before);
  });
});

describe("Logo", () => {
  it("renders an accessible nitid logo", () => {
    render(<Logo className="logo" />);
    expect(screen.getByLabelText("nitid")).toBeTruthy();
  });
});
