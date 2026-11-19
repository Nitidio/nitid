import { useMemo } from "react";
import { filterDetections } from "../local/filter";
import type { ExportMenuProps as SlotProps, ViewFilter } from "../local/results-model";
import { ExportMenu } from "./ExportMenu";
import type { ExportResults } from "./model";

/** "Export what I see": keep the detections that pass the on-screen confidence and class filter. */
export function applyViewFilter(results: ExportResults, filter: ViewFilter): ExportResults {
  return { ...results, detections: filterDetections(results.detections, filter) };
}

/**
 * Adapter between the results views (which pass `RunResults`, the on-screen filter and the decoded
 * image) and `ExportMenu` (which takes `ExportResults`). The conversion goes through the shared
 * mappers: `RunResults.toExport` is `fromServerRun` for server runs and `fromEngineResult` for
 * browser runs.
 */
export function ResultsExportMenu({ results, filter, selectedItemId, image }: SlotProps) {
  const exportResults = useMemo(() => {
    if (!results.toExport) return null;
    return applyViewFilter(results.toExport(selectedItemId), filter);
  }, [results, filter, selectedItemId]);

  if (!exportResults) return null;
  return <ExportMenu results={exportResults} {...(image ? { image } : {})} />;
}
