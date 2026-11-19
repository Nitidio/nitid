/**
 * CSV: one row per detection (RFC 4180: CRLF line ends, fields with `,` `"`
 * CR or LF are quoted, quotes doubled).
 *
 * Text cells that a spreadsheet could read as a formula (leading `=`, `+`,
 * `-`, `@`, TAB or CR) get a leading apostrophe ("CSV injection": class names
 * come from user-supplied models). Numeric cells are never prefixed.
 */
import type { ExportResults } from "./model";

export const CSV_COLUMNS = [
  "image",
  "index",
  "class_id",
  "class_name",
  "confidence",
  "x1",
  "y1",
  "x2",
  "y2",
  "width",
  "height",
  "has_mask",
  "mask_pixels",
] as const;

export interface CsvOptions {
  /** Field separator; use ";" for Excel with a Spanish/European locale. Default ",". */
  delimiter?: "," | ";" | "\t";
  /** Prefix a UTF-8 byte-order mark so Excel detects Unicode. Default false. */
  bom?: boolean;
}

function cell(value: string, delimiter: string, isText: boolean): string {
  let v = value;
  if (isText && /^[=+\-@\t\r]/.test(v)) v = `'${v}`;
  if (v.includes('"') || v.includes(delimiter) || v.includes("\n") || v.includes("\r")) {
    return `"${v.replaceAll('"', '""')}"`;
  }
  return v;
}

const num = (v: number, digits: number): string => String(Number(v.toFixed(digits)));

function maskPixels(data: Uint8Array | undefined): string {
  if (!data) return "";
  let n = 0;
  for (let i = 0; i < data.length; i++) if (data[i]) n++;
  return String(n);
}

export function formatCsv(results: ExportResults, options: CsvOptions = {}): string {
  const delimiter = options.delimiter ?? ",";
  const rows: string[] = [CSV_COLUMNS.join(delimiter)];
  results.detections.forEach((d, index) => {
    const fields: Array<[string, boolean]> = [
      [results.image.name, true],
      [String(index), false],
      [String(d.classId), false],
      [d.name, true],
      [num(d.confidence, 6), false],
      [num(d.box.x1, 2), false],
      [num(d.box.y1, 2), false],
      [num(d.box.x2, 2), false],
      [num(d.box.y2, 2), false],
      [num(d.box.x2 - d.box.x1, 2), false],
      [num(d.box.y2 - d.box.y1, 2), false],
      [d.hasMask ? "true" : "false", false],
      [maskPixels(d.mask?.data), false],
    ];
    rows.push(fields.map(([v, text]) => cell(v, delimiter, text)).join(delimiter));
  });
  return `${options.bom ? "﻿" : ""}${rows.join("\r\n")}\r\n`;
}
