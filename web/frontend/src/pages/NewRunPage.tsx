import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { createRun } from "../api/runs";
import { FileDropzone } from "../components/FileDropzone";
import { ModelSelector } from "../components/ModelSelector";
import { ParameterForm } from "../components/ParameterForm";
import type { RunCreateParams } from "../types";

const VIDEO_EXTS = new Set([".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"]);

function hasVideo(files: File[]): boolean {
  return files.some((f) => VIDEO_EXTS.has(f.name.slice(f.name.lastIndexOf(".")).toLowerCase()));
}

export function NewRunPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [files, setFiles] = useState<File[]>([]);
  const [modelName, setModelName] = useState("");
  const [params, setParams] = useState<RunCreateParams>({
    model_name: "",
    conf: 0.5,
    imgsz: 640,
    classes: null,
    frame_step: 30,
  });

  const { mutate: submit, isPending, error } = useMutation({
    mutationFn: ({ p, fs }: { p: RunCreateParams; fs: File[] }) => createRun(p, fs),
    onSuccess: (run) => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      navigate(`/runs/${run.id}`);
    },
  });

  function handleModelChange(m: string) {
    setModelName(m);
    setParams((p) => ({ ...p, model_name: m }));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!modelName) return alert("Select a model first.");
    if (files.length === 0) return alert("Add at least one file.");
    submit({ p: { ...params, model_name: modelName }, fs: files });
  }

  const errMsg = (error as any)?.response?.data?.detail ?? (error as any)?.message;

  return (
    <div className="page">
      <h2>New Run</h2>
      <form onSubmit={handleSubmit} className="new-run-form">
        <section>
          <h3>1. Select model</h3>
          <ModelSelector value={modelName} onChange={handleModelChange} />
        </section>

        <section>
          <h3>2. Configure parameters</h3>
          <ParameterForm
            modelName={modelName}
            params={params}
            hasVideo={hasVideo(files)}
            onChange={setParams}
          />
        </section>

        <section>
          <h3>3. Upload files</h3>
          <FileDropzone files={files} onChange={setFiles} />
        </section>

        {errMsg && <p className="error-msg">{errMsg}</p>}

        <button type="submit" className="btn-primary" disabled={isPending}>
          {isPending ? "Submitting..." : "Run detection"}
        </button>
      </form>
    </div>
  );
}
