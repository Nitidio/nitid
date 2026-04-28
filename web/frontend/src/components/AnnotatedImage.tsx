import { useState } from "react";
import { fileUrl } from "../api/runs";

interface Props {
  runId: number;
  filename: string;
  label?: string;
  onClick?: () => void;
}

export function AnnotatedImage({ runId, filename, label, onClick }: Props) {
  const [error, setError] = useState(false);
  const src = fileUrl(runId, filename);

  if (error) {
    return (
      <div className="annotated-image-placeholder">
        <span>Image unavailable</span>
      </div>
    );
  }

  return (
    <div className="annotated-image-wrapper" onClick={onClick} style={{ cursor: onClick ? "pointer" : "default" }}>
      <img
        src={src}
        alt={label ?? filename}
        onError={() => setError(true)}
        style={{ width: "100%", borderRadius: 4 }}
      />
      {label && <div className="annotated-image-label">{label}</div>}
    </div>
  );
}
