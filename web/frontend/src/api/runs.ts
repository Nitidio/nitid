import client from "./client";
import type { Run, RunDetail, RunCreateParams } from "../types";

export async function listRuns(skip = 0, limit = 50): Promise<Run[]> {
  const { data } = await client.get<Run[]>("/runs", { params: { skip, limit } });
  return data;
}

export async function getRun(id: number): Promise<RunDetail> {
  const { data } = await client.get<RunDetail>(`/runs/${id}`);
  return data;
}

export async function createRun(params: RunCreateParams, files: File[]): Promise<Run> {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  formData.append("params", JSON.stringify(params));
  const { data } = await client.post<Run>("/runs", formData);
  return data;
}

export async function deleteRun(id: number): Promise<void> {
  await client.delete(`/runs/${id}`);
}

export function fileUrl(runId: number, filename: string): string {
  const base = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
  const token = localStorage.getItem("access_token") ?? "";
  return `${base}/files/${runId}/${filename}?token=${token}`;
}
