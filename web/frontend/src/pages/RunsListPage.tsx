import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { deleteRun, listRuns } from "../api/runs";
import { RunStatusBadge } from "../components/RunStatusBadge";

export function RunsListPage() {
  const queryClient = useQueryClient();
  const { data: runs = [], isLoading } = useQuery({
    queryKey: ["runs"],
    queryFn: () => listRuns(),
    refetchInterval: (query) => {
      const runs = query.state.data;
      const hasPending = runs?.some((r) => r.status === "pending" || r.status === "running");
      return hasPending ? 2000 : false;
    },
  });

  const { mutate: remove } = useMutation({
    mutationFn: deleteRun,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runs"] }),
  });

  if (isLoading) return <div className="loading">Loading runs...</div>;

  return (
    <div className="page">
      <div className="page-header">
        <h2>Runs</h2>
        <Link to="/runs/new" className="btn-primary">+ New Run</Link>
      </div>
      {runs.length === 0 ? (
        <p>No runs yet. <Link to="/runs/new">Start one</Link>.</p>
      ) : (
        <table className="runs-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Model</th>
              <th>Status</th>
              <th>Input</th>
              <th>Items</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.model_name}</td>
                <td><RunStatusBadge status={r.status} /></td>
                <td>{r.input_type}</td>
                <td>{r.item_count}</td>
                <td>{new Date(r.created_at).toLocaleString()}</td>
                <td>
                  <Link to={`/runs/${r.id}`} className="btn-sm">View</Link>
                  <button
                    className="btn-sm btn-danger"
                    onClick={() => { if (confirm("Delete this run?")) remove(r.id); }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
