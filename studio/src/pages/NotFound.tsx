import { Link } from "react-router";
import { usePageTitle } from "../use-page-title";

export function NotFound() {
  usePageTitle("Not found");
  return (
    <>
      <h1>Page not found</h1>
      <p>There is nothing at this address.</p>
      <p>
        <Link to="/">Back to start</Link>
      </p>
    </>
  );
}
