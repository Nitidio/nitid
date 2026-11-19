import { type ComponentType } from "react";
import { Link } from "react-router";
import { useBackend } from "../backend-context";
import { LocalPanel } from "../local/LocalPanel";
import type { ExportMenuProps } from "../local/results-model";
import { usePageTitle } from "../use-page-title";

/** `/run/local`: the workbench when a local server is present, an explanation otherwise. */
export function RunLocal({ ExportMenu }: { ExportMenu?: ComponentType<ExportMenuProps> }) {
  usePageTitle("Local mode");
  const backend = useBackend();

  if (backend === null) return <p role="status">Checking for a local server...</p>;
  if (backend.mode !== "local") {
    return (
      <>
        <h1>Local mode</h1>
        <p role="alert" className="notice">
          No local server found. Start it with <code>nitid studio</code>, or use the in-browser mode.
        </p>
        <p>
          <Link to="/">Back to start</Link>
        </p>
      </>
    );
  }
  return <LocalPanel {...(ExportMenu ? { ExportMenu } : {})} />;
}
