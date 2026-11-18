# Nitid Studio (frontend)

Vite + React + TypeScript single-page app for Nitid Studio, a workbench that
runs models either on this computer through the Nitid Python server ("Local")
or inside the browser with ONNX Runtime Web ("Browser").

```bash
cd studio
pnpm install --frozen-lockfile
pnpm run dev      # http://localhost:5173, proxies /api to 127.0.0.1:8000
pnpm run check    # lint + typecheck + test + build
```

Requires Node.js >= 22.13 and the pnpm version in `package.json`
(`packageManager`). Full guide: [`docs/studio-development.md`](../docs/studio-development.md).
