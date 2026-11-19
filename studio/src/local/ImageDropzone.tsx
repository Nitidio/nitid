import { useId, useRef, useState, type DragEvent } from "react";
import { checkFiles, formatBytes, type FileProblem } from "./files";

export interface ImageDropzoneProps {
  files: readonly File[];
  onChange: (files: File[]) => void;
  disabled?: boolean;
  /** Several images (default) or exactly one: a new pick then replaces the old one. */
  multiple?: boolean;
  /** Help text under the button; the default describes the server mode. */
  hint?: string;
}

const DEFAULT_HINT = "JPEG, PNG or WebP. Up to 25 MB each. Images are sent only to the server on this computer.";

export function ImageDropzone({ files, onChange, disabled, multiple = true, hint = DEFAULT_HINT }: ImageDropzoneProps) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [problems, setProblems] = useState<FileProblem[]>([]);

  function add(list: FileList | null) {
    if (!list || disabled) return;
    const { accepted, rejected } = checkFiles([...list], multiple ? files : []);
    setProblems(rejected);
    if (accepted.length === 0) return;
    onChange(multiple ? [...files, ...accepted] : [accepted[0]!]);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    add(e.dataTransfer.files);
  }

  return (
    <div className="field">
      <span id={`${id}-label`} className="label">
        Images
      </span>
      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p>{multiple ? "Drop images here, or" : "Drop an image here, or"}</p>
        <input
          ref={input}
          id={`${id}-input`}
          className="visually-hidden-input"
          type="file"
          accept="image/*"
          multiple={multiple}
          disabled={disabled}
          aria-labelledby={`${id}-label`}
          aria-describedby={`${id}-hint`}
          onChange={(e) => {
            add(e.target.files);
            e.target.value = "";
          }}
        />
        <label htmlFor={`${id}-input`} className="button-like" aria-disabled={disabled}>
          {multiple ? "Choose images" : "Choose an image"}
        </label>
        <p id={`${id}-hint`} className="hint">
          {hint}
        </p>
      </div>

      {problems.length > 0 && (
        <ul role="alert" className="problems">
          {problems.map((p, i) => (
            <li key={`${p.name}-${i}`}>
              {p.name}: {p.reason}
            </li>
          ))}
        </ul>
      )}

      {files.length > 0 && (
        <ul className="file-list" aria-label="Selected images">
          {files.map((f, i) => (
            <li key={`${f.name}-${f.size}-${f.lastModified}-${i}`}>
              <span>
                {f.name} <span className="hint">({formatBytes(f.size)})</span>
              </span>
              <button
                type="button"
                className="secondary small"
                disabled={disabled}
                aria-label={`Remove ${f.name}`}
                onClick={() => onChange(files.filter((_, j) => j !== i))}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
