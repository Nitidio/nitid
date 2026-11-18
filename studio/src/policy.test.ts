import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

// Static guards for the security/CSP rules in docs/studio-development.md.
const root = resolve(__dirname, "..");

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? files(p) : [p];
  });
}

const sources = files(join(root, "src")).filter(
  (f) => /\.(ts|tsx|css)$/.test(f) && !/\.test\.tsx?$/.test(f),
);

describe("source policy", () => {
  it("never references localStorage in application code", () => {
    for (const f of sources) expect(readFileSync(f, "utf8"), f).not.toMatch(/localStorage/);
  });

  it("has no external http(s) URLs in code or styles (same-origin only)", () => {
    for (const f of sources) expect(readFileSync(f, "utf8"), f).not.toMatch(/https?:\/\//);
  });

  it("index.html has no inline scripts and no external resources", () => {
    const html = readFileSync(join(root, "index.html"), "utf8");
    expect(html).toContain('lang="en"');
    expect(html).toMatch(/<title>[^<]+<\/title>/);
    expect(html).not.toMatch(/https?:\/\//);
    for (const tag of html.match(/<script\b[^>]*>/g) ?? []) expect(tag).toMatch(/\bsrc=/);
    expect(html).not.toMatch(/\son[a-z]+=/i);
    expect(html).not.toMatch(/<style\b/);
  });
});
