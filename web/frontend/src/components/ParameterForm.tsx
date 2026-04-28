import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getModelClasses } from "../api/models";
import type { RunCreateParams } from "../types";

interface Props {
  modelName: string;
  params: RunCreateParams;
  hasVideo: boolean;
  onChange: (p: RunCreateParams) => void;
}

export function ParameterForm({ modelName, params, hasVideo, onChange }: Props) {
  const [classInput, setClassInput] = useState("");

  const { data: classMap = {} } = useQuery({
    queryKey: ["model-classes", modelName],
    queryFn: () => getModelClasses(modelName),
    enabled: !!modelName,
  });

  const classEntries = Object.entries(classMap).sort((a, b) => Number(a[0]) - Number(b[0]));

  useEffect(() => {
    setClassInput(params.classes ? params.classes.join(",") : "");
  }, [modelName]);

  function handleClassInput(val: string) {
    setClassInput(val);
    const ids = val
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n));
    onChange({ ...params, classes: ids.length > 0 ? ids : null });
  }

  return (
    <div className="param-form">
      <label>
        Confidence threshold: <strong>{params.conf.toFixed(2)}</strong>
        <input
          type="range"
          min={0.01}
          max={1}
          step={0.01}
          value={params.conf}
          onChange={(e) => onChange({ ...params, conf: parseFloat(e.target.value) })}
        />
      </label>

      <label>
        Image size:
        <select
          value={params.imgsz}
          onChange={(e) => onChange({ ...params, imgsz: parseInt(e.target.value) })}
        >
          {[320, 480, 640, 960, 1280].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </label>

      {classEntries.length > 0 && (
        <label>
          Filter classes (comma-separated IDs, empty = all):
          <input
            type="text"
            placeholder="e.g. 0,1,2"
            value={classInput}
            onChange={(e) => handleClassInput(e.target.value)}
          />
          <div className="class-hint">
            {classEntries.slice(0, 20).map(([id, name]) => (
              <span key={id} className="class-chip">{id}: {name}</span>
            ))}
            {classEntries.length > 20 && <span>…+{classEntries.length - 20} more</span>}
          </div>
        </label>
      )}

      {hasVideo && (
        <label>
          Frame step (sample every N frames): <strong>{params.frame_step}</strong>
          <input
            type="number"
            min={1}
            max={300}
            value={params.frame_step}
            onChange={(e) => onChange({ ...params, frame_step: Math.max(1, parseInt(e.target.value) || 1) })}
          />
        </label>
      )}
    </div>
  );
}
