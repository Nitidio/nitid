import type { Detection } from "../types";

export function DetectionTable({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return <p style={{ color: "#888" }}>No detections.</p>;
  }
  return (
    <table className="detection-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Class</th>
          <th>Confidence</th>
          <th>x1</th>
          <th>y1</th>
          <th>x2</th>
          <th>y2</th>
        </tr>
      </thead>
      <tbody>
        {detections.map((d, i) => (
          <tr key={i}>
            <td>{i + 1}</td>
            <td>{d.name}</td>
            <td>{(d.confidence * 100).toFixed(1)}%</td>
            <td>{Math.round(d.box.x1)}</td>
            <td>{Math.round(d.box.y1)}</td>
            <td>{Math.round(d.box.x2)}</td>
            <td>{Math.round(d.box.y2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
