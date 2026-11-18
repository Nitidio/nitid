import { useBackend } from "../backend-context";
import { usePageTitle } from "../use-page-title";

export function About() {
  usePageTitle("About");
  const backend = useBackend();

  return (
    <>
      <h1>About</h1>
      <p>
        Nitid Studio is a small workbench for trying Nitid models. It can run a model on this
        computer through a local Python server, or entirely inside your browser.
      </p>
      <dl>
        <dt>Deployment</dt>
        <dd>
          {backend === null && "Detecting..."}
          {backend?.mode === "local" && "Local server (local and browser modes)"}
          {backend?.mode === "hosted" && "Static site (browser mode only)"}
        </dd>
        {backend?.health && (
          <>
            <dt>Nitid version</dt>
            <dd>{backend.health.nitid_version}</dd>
            <dt>API version</dt>
            <dd>{backend.health.api_version}</dd>
          </>
        )}
      </dl>
    </>
  );
}
