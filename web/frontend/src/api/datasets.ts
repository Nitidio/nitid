import client from "./client";

export async function convertDataset(
  archive: File,
  target: "coco" | "yolo",
  config?: string,
): Promise<Blob> {
  const form = new FormData();
  form.append("archive", archive);
  form.append("target", target);
  if (config) form.append("config", config);
  const response = await client.post("/datasets/convert", form, { responseType: "blob" });
  return response.data;
}
