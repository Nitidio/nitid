/** Per-image limit of the contract (`x-limits.max_image_bytes`). The server stays authoritative. */
export const MAX_IMAGE_BYTES = 26_214_400;

export interface FileProblem {
  name: string;
  reason: string;
}

export interface FileCheck {
  accepted: File[];
  rejected: FileProblem[];
}

/** Client-side pre-flight: images only, within the size limit, no duplicates of what is queued. */
export function checkFiles(incoming: readonly File[], existing: readonly File[] = []): FileCheck {
  const accepted: File[] = [];
  const rejected: FileProblem[] = [];
  const sig = (f: File) => `${f.name}\u0000${f.size}\u0000${f.lastModified}`;
  const seen = new Set(existing.map(sig));
  for (const file of incoming) {
    if (!file.type.startsWith("image/")) {
      rejected.push({ name: file.name, reason: "not an image" });
    } else if (file.size > MAX_IMAGE_BYTES) {
      rejected.push({ name: file.name, reason: "larger than 25 MB" });
    } else if (file.size === 0) {
      rejected.push({ name: file.name, reason: "empty file" });
    } else if (seen.has(sig(file))) {
      rejected.push({ name: file.name, reason: "already added" });
    } else {
      seen.add(sig(file));
      accepted.push(file);
    }
  }
  return { accepted, rejected };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
