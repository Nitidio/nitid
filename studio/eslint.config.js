import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/", "dist-hosted/", "dist-hosted-site/", "dist-e2e-harness/", "coverage/", "node_modules/", ".vite/", "public/ort/"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: { globals: { ...globals.browser } },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
    },
  },
  {
    files: ["e2e/harness/*.ts"],
    languageOptions: { globals: { ...globals.browser } },
  },
  {
    files: ["vite.config.ts", "eslint.config.js", "scripts/**/*.mjs", "e2e/*.mjs", "e2e/harness/vite.config.ts"],
    languageOptions: { globals: { ...globals.node } },
  },
);
