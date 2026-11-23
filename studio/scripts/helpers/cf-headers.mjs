// A small emulation of Cloudflare's `_headers` file (Workers static assets), used by the hosted smoke
// test so that the test server sends exactly the headers the production file declares.
// Documented behaviour (https://developers.cloudflare.com/workers/static-assets/headers/):
//   - a block is a URL pattern line followed by indented `Name: value` lines; `#` starts a comment;
//   - a request inherits the headers of every matching block, in file order;
//   - the same header set twice is joined with ", "; `! Name` removes the header (defaults included);
//   - `*` is a greedy splat (one per pattern); `:name` matches anything but "/".
// Not emulated: absolute-URL patterns (https://host/path), which this project does not use.

/** Parse the file into `[{ pattern, set: [[name, value|null]...] }]`; a `null` value is a detach (`! Name`). */
export function parseHeadersFile(text) {
  const rules = [];
  let current = null;
  text.split(/\r?\n/).forEach((raw, index) => {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim() || line.trim().startsWith("#")) return;
    if (line.length > 2000) throw new Error(`_headers line ${index + 1} is longer than 2000 characters`);
    if (!/^\s/.test(line)) {
      current = { pattern: line.trim(), set: [] };
      if (!current.pattern.startsWith("/")) throw new Error(`_headers line ${index + 1}: pattern must start with "/"`);
      if ((current.pattern.match(/\*/g) ?? []).length > 1) throw new Error(`_headers line ${index + 1}: one splat per pattern`);
      rules.push(current);
      return;
    }
    if (!current) throw new Error(`_headers line ${index + 1}: header before any URL pattern`);
    const body = line.trim();
    if (body.startsWith("! ")) {
      current.set.push([body.slice(2).trim(), null]);
      return;
    }
    const colon = body.indexOf(":");
    if (colon <= 0) throw new Error(`_headers line ${index + 1}: expected "Name: value"`);
    current.set.push([body.slice(0, colon).trim(), body.slice(colon + 1).trim()]);
  });
  if (rules.length > 100) throw new Error(`_headers has ${rules.length} rules; Cloudflare allows 100`);
  return rules;
}

/** Regular expression that a URL pattern stands for. */
export function patternToRegExp(pattern) {
  const source = pattern
    .split(/(\*|:[A-Za-z]\w*)/)
    .map((part) => {
      if (part === "*") return "(.*)";
      if (/^:[A-Za-z]\w*$/.test(part)) return "([^/]+)";
      return part.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
    })
    .join("");
  return new RegExp(`^${source}$`);
}

/** The headers (lower-case name -> value) that `rules` add to a response for `pathname`. */
export function headersFor(rules, pathname) {
  const out = new Map();
  for (const rule of rules) {
    if (!patternToRegExp(rule.pattern).test(pathname)) continue;
    for (const [name, value] of rule.set) {
      const key = name.toLowerCase();
      if (value === null) {
        out.delete(key);
      } else if (out.has(key)) {
        out.set(key, `${out.get(key)}, ${value}`);
      } else {
        out.set(key, value);
      }
    }
  }
  return Object.fromEntries(out);
}
