import type { DescribedError } from "./errors";

export function ErrorNotice({ error, title }: { error: DescribedError; title?: string }) {
  return (
    <div role="alert" className="notice local-error">
      {title && <strong>{title} </strong>}
      <span>{error.message}</span>
      {error.code && <code className="error-code"> [{error.code}]</code>}
      {error.hint && <p className="error-hint">{error.hint}</p>}
    </div>
  );
}
