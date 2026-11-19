import type { Model, Task } from "../api/types";

export const modelKey = (m: Pick<Model, "id" | "task">): string => `${m.task}:${m.id}`;

export const TASK_LABEL: Record<Task, string> = {
  detect: "Object detection",
  segment: "Instance segmentation",
};

export function modelLabel(m: Model): string {
  const size = m.size ? ` (${m.size.toUpperCase()})` : "";
  const src = m.source === "local" ? " [local file]" : "";
  return `${m.id}${size}${src}`;
}

export function groupByTask(models: readonly Model[]): { task: Task; models: Model[] }[] {
  const order: Task[] = ["detect", "segment"];
  return order
    .map((task) => ({ task, models: models.filter((m) => m.task === task) }))
    .filter((g) => g.models.length > 0);
}

export function findModel(models: readonly Model[], key: string): Model | undefined {
  return models.find((m) => modelKey(m) === key);
}

/** Preferred starting point: first downloaded detect model, else the first model listed. */
export function defaultModelKey(models: readonly Model[]): string | null {
  const pick =
    models.find((m) => m.downloaded && m.task === "detect") ??
    models.find((m) => m.downloaded) ??
    models[0];
  return pick ? modelKey(pick) : null;
}
