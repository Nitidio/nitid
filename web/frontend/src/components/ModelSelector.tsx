import { useQuery } from "@tanstack/react-query";
import { listModels } from "../api/models";

interface Props {
  value: string;
  onChange: (model: string) => void;
}

export function ModelSelector({ value, onChange }: Props) {
  const { data: models = [], isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
  });

  if (isLoading) return <select disabled><option>Loading...</option></select>;
  if (models.length === 0) return <p style={{ color: "#ef4444" }}>No models found in models/ directory.</p>;

  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">-- select model --</option>
      {models.map((m) => (
        <option key={m} value={m}>{m}</option>
      ))}
    </select>
  );
}
