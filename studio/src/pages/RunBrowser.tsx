import { BrowserPanel } from "../browser/BrowserPanel";
import { usePageTitle } from "../use-page-title";

/** `/run/browser`: in-browser inference; works with or without a local server. */
export function RunBrowser() {
  usePageTitle("Browser mode");
  return <BrowserPanel />;
}
