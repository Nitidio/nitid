import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { parseHostedCatalog } from "./catalog";

/**
 * Contract between the Python bundle publisher (tools/publish_models.py) and the browser parser.
 *
 * By default this reads the committed fixture, which tests/unit/test_publish_models.py proves is
 * byte-identical to what `build_catalogue` + `dump_json` produce. Set NITID_HOSTED_CATALOG to the
 * models.json of a real bundle to check that one instead (run from studio/):
 *
 *     NITID_HOSTED_CATALOG=/path/to/bundle/models.json pnpm exec vitest run hosted-bundle
 */
const path = process.env.NITID_HOSTED_CATALOG ?? resolve(process.cwd(), "src/browser/__fixtures__/hosted-models.json");
const catalogue: unknown = JSON.parse(readFileSync(path, "utf8"));
const BASE = "https://app.example.test/models/";

describe("catalogue written by tools/publish_models.py", () => {
  it("is accepted by parseHostedCatalog", () => {
    const models = parseHostedCatalog(catalogue, BASE);
    expect(models.length).toBeGreaterThan(0);
    for (const m of models) {
      expect(m.origin).toBe("hosted");
      expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it("resolves every relative url against the base, on the same origin", () => {
    for (const m of parseHostedCatalog(catalogue, BASE)) {
      expect(new URL(m.url).origin).toBe(new URL(BASE).origin);
      expect(m.url.startsWith(BASE)).toBe(true);
      expect(m.url.endsWith(".onnx")).toBe(true);
    }
  });

  it("keeps the labels the publisher wrote", () => {
    const models = parseHostedCatalog(catalogue, BASE);
    const raw = (catalogue as { models: { label: string }[] }).models;
    expect(models.map((m) => m.label)).toEqual(raw.map((m) => m.label));
  });
});
