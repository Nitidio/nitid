import { useEffect } from "react";

const BASE = "Nitid Studio";

/** Sets `document.title` so each route has a distinct, descriptive title. */
export function usePageTitle(title?: string): void {
  useEffect(() => {
    document.title = title ? `${title} - ${BASE}` : BASE;
  }, [title]);
}
