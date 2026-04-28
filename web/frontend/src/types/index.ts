export interface User {
  id: number;
  username: string;
  created_at: string;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  box: BoundingBox;
  confidence: number;
  class: number;
  name: string;
}

export interface RunItem {
  id: number;
  source_path: string;
  result_snapshot_path: string | null;
  detections: Detection[];
  frame_idx: number | null;
}

export type RunStatus = "pending" | "running" | "done" | "failed";

export interface Run {
  id: number;
  model_name: string;
  conf: number;
  imgsz: number;
  classes: number[] | null;
  frame_step: number;
  status: RunStatus;
  input_type: string;
  created_at: string;
  completed_at: string | null;
  error_msg: string | null;
  item_count: number;
}

export interface RunDetail extends Run {
  items: RunItem[];
}

export interface RunCreateParams {
  model_name: string;
  conf: number;
  imgsz: number;
  classes: number[] | null;
  frame_step: number;
}
