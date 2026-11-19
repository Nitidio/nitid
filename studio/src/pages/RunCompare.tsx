import { Link } from "react-router";
import { useBackend } from "../backend-context";
import { ComparePanel } from "../compare/ComparePanel";
import { usePageTitle } from "../use-page-title";

/** `/compare`: needs the local server (one side runs there); the hosted site explains why it is absent. */
export function RunCompare() {
  usePageTitle("Compare");
  const backend = useBackend();
  if (backend === null) return <p role="status">Checking for a local server...</p>;
  if (backend.mode !== "local") {
    return (
      <>
        <h1>Compare</h1>
        <p role="alert" className="notice">
          Comparing needs the local server, because one side runs there. Start it with <code>nitid studio</code>, or
          use the in-browser mode on its own.
        </p>
        <p>
          <Link to="/run/browser">Browser mode</Link> - <Link to="/">Back to start</Link>
        </p>
      </>
    );
  }
  return <ComparePanel />;
}
