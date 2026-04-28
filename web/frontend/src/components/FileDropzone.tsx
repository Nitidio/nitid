import { useRef, useState } from "react";

interface Props {
  files: File[];
  onChange: (files: File[]) => void;
}

export function FileDropzone({ files, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(incoming: FileList | null) {
    if (!incoming) return;
    onChange(Array.from(incoming));
  }

  return (
    <div
      className={`dropzone${dragging ? " dragging" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="image/*,video/*"
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />
      {files.length === 0 ? (
        <p>Drop images / video here, or click to browse</p>
      ) : (
        <ul className="file-list">
          {files.map((f, i) => (
            <li key={i}>
              {f.type.startsWith("image/") && (
                <img
                  src={URL.createObjectURL(f)}
                  alt={f.name}
                  style={{ width: 48, height: 48, objectFit: "cover", marginRight: 8, borderRadius: 4 }}
                />
              )}
              <span>{f.name}</span>
              <span style={{ color: "#888", marginLeft: 8 }}>({(f.size / 1024).toFixed(0)} KB)</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
