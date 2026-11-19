import { useEffect, useId, useMemo, useState, type ComponentType } from "react";
import { DetectionCanvas } from "./DetectionCanvas";
import { clampConfidence, defaultFilter, filterDetections, setAllClasses, summarizeClasses, toggleClass } from "./filter";
import { itemTotalMs, type ExportMenuProps, type ResultItem, type RunResults, type ViewFilter } from "./results-model";

export interface ResultsViewProps {
  results: RunResults;
  /** Smallest confidence the source can supply (the run's `conf`); the slider cannot go below it. */
  floorConfidence?: number;
  initialConfidence?: number;
  /** Loads the original image of an item. Returns null when unavailable (a placeholder is drawn). */
  loadImage: (item: ResultItem) => Promise<ImageBitmap | null>;
  /** Slot for the export controls; the unified UI plugs in `ResultsExportMenu`. */
  ExportMenu?: ComponentType<ExportMenuProps>;
}

function fmtMs(n: number): string {
  return n >= 100 ? `${Math.round(n)} ms` : `${n.toFixed(1)} ms`;
}

export function ResultsView({
  results,
  floorConfidence = 0,
  initialConfidence = 0.25,
  loadImage,
  ExportMenu,
}: ResultsViewProps) {
  const id = useId();
  const [filter, setFilter] = useState<ViewFilter>(() =>
    defaultFilter(clampConfidence(initialConfidence, floorConfidence)),
  );
  const [selectedId, setSelectedId] = useState(results.items[0]?.id ?? 0);
  const [showLabels, setShowLabels] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
  const [images, setImages] = useState<Record<number, ImageBitmap | null>>({});

  const item = results.items.find((i) => i.id === selectedId) ?? results.items[0];

  useEffect(() => {
    if (!item || item.id in images) return;
    let cancelled = false;
    loadImage(item)
      .catch(() => null)
      .then((bmp) => {
        if (!cancelled) setImages((m) => ({ ...m, [item.id]: bmp }));
      });
    return () => {
      cancelled = true;
    };
  }, [item, images, loadImage]);

  useEffect(
    () => () => {
      for (const bmp of Object.values(images)) bmp?.close?.();
    },
    // Close bitmaps only when the whole view goes away or the run changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [results.runId],
  );

  const visible = useMemo(() => (item ? filterDetections(item.detections, filter) : []), [item, filter]);
  const classes = useMemo(() => summarizeClasses(results.items, filter), [results.items, filter]);
  const anyMask = results.task === "segment" || visible.some((d) => d.hasMask);

  if (!item) return <p>The run finished without any images.</p>;

  const pct = Math.round(filter.minConfidence * 100);
  const canvasLabel = `${item.name}: ${visible.length} detection${visible.length === 1 ? "" : "s"} shown`;

  return (
    <section aria-labelledby={`${id}-h`} className="results">
      <h2 id={`${id}-h`}>Results</h2>
      <p className="hint">
        {results.model} ({results.task}) on {results.backend}
        {results.device ? ` / ${results.device}` : ""}
        {results.wallClockMs !== undefined && <> - total {fmtMs(results.wallClockMs)} including upload</>}
      </p>

      {results.items.length > 1 && (
        <div className="field">
          <label htmlFor={`${id}-item`}>Image</label>
          <select id={`${id}-item`} value={item.id} onChange={(e) => setSelectedId(Number(e.target.value))}>
            {results.items.map((i) => (
              <option key={i.id} value={i.id}>
                {i.name} ({i.detections.length})
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="results-grid">
        <div>
          <DetectionCanvas
            image={images[item.id] ?? null}
            width={item.width}
            height={item.height}
            detections={visible}
            showLabels={showLabels}
            showMasks={showMasks}
            label={canvasLabel}
          />
          <p role="status" className="hint">
            {visible.length} of {item.detections.length} detections shown.{" "}
            Preprocess {fmtMs(item.speed.preprocess)}, inference {fmtMs(item.speed.inference)}, postprocess{" "}
            {fmtMs(item.speed.postprocess)} (total {fmtMs(itemTotalMs(item))}).
          </p>
        </div>

        <div className="controls">
          <div className="field">
            <label htmlFor={`${id}-conf`}>Minimum confidence: {pct}%</label>
            <input
              id={`${id}-conf`}
              type="range"
              min={Math.round(floorConfidence * 100)}
              max={100}
              step={1}
              value={pct}
              aria-valuetext={`${pct} percent`}
              onChange={(e) =>
                setFilter((f) => ({ ...f, minConfidence: clampConfidence(Number(e.target.value) / 100, floorConfidence) }))
              }
            />
          </div>

          <div className="field">
            <label>
              <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> Show
              labels
            </label>
            {anyMask && (
              <label>
                <input type="checkbox" checked={showMasks} onChange={(e) => setShowMasks(e.target.checked)} /> Show
                masks
              </label>
            )}
          </div>

          <fieldset className="field">
            <legend>Classes</legend>
            {classes.length === 0 ? (
              <p className="hint">No classes were detected.</p>
            ) : (
              <>
                <div className="inline-buttons">
                  <button
                    type="button"
                    className="secondary small"
                    onClick={() =>
                      setFilter((f) =>
                        setAllClasses(
                          f,
                          classes.map((c) => c.classId),
                          true,
                        ),
                      )
                    }
                  >
                    All
                  </button>
                  <button
                    type="button"
                    className="secondary small"
                    onClick={() =>
                      setFilter((f) =>
                        setAllClasses(
                          f,
                          classes.map((c) => c.classId),
                          false,
                        ),
                      )
                    }
                  >
                    None
                  </button>
                </div>
                <ul className="class-list">
                  {classes.map((c) => (
                    <li key={c.classId}>
                      <label>
                        <input
                          type="checkbox"
                          checked={!c.hidden}
                          onChange={() => setFilter((f) => toggleClass(f, c.classId))}
                        />{" "}
                        {c.name} <span className="hint">({c.count})</span>
                      </label>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </fieldset>

          {ExportMenu && <ExportMenu results={results} filter={filter} selectedItemId={item.id} image={images[item.id] ?? null} />}
        </div>
      </div>

      {anyMask && (
        <p className="hint">
          Mask outlines are not provided by the server API yet; instances with a mask are marked with a tinted box.
        </p>
      )}

      <table className="detections">
        <caption>Detections in {item.name}</caption>
        <thead>
          <tr>
            <th scope="col">Class</th>
            <th scope="col">Confidence</th>
            <th scope="col">Box (x1, y1, x2, y2 px)</th>
            {anyMask && <th scope="col">Mask</th>}
          </tr>
        </thead>
        <tbody>
          {visible.length === 0 ? (
            <tr>
              <td colSpan={anyMask ? 4 : 3}>Nothing above the current threshold.</td>
            </tr>
          ) : (
            visible.map((d, i) => (
              <tr key={i}>
                <td>{d.name}</td>
                <td>{(d.confidence * 100).toFixed(1)}%</td>
                <td>
                  {Math.round(d.box.x1)}, {Math.round(d.box.y1)}, {Math.round(d.box.x2)}, {Math.round(d.box.y2)}
                </td>
                {anyMask && <td>{d.hasMask ? "yes" : "no"}</td>}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </section>
  );
}
