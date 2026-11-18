import { Link, useParams } from "react-router";
import { usePageTitle } from "../use-page-title";

/** Stand-in for the workbench, which arrives in a later work package. */
export function RunPlaceholder() {
  const { mode } = useParams();
  const valid = mode === "local" || mode === "browser";
  usePageTitle(valid ? `${mode === "local" ? "Local" : "Browser"} mode` : "Not found");

  return (
    <>
      <h1>{valid ? `${mode === "local" ? "Local" : "Browser"} mode` : "Page not found"}</h1>
      <p>{valid ? "The workbench is not available in this build yet." : "There is nothing at this address."}</p>
      <p>
        <Link to="/">Back to start</Link>
      </p>
    </>
  );
}
