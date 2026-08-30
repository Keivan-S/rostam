import type { ReactNode } from 'react';
import { AlertTriangle, KeyRound, Inbox, RefreshCw } from 'lucide-react';
import { ApiError } from '../api/client';
import { Card, Spinner } from './primitives';

export function ErrorCard({
  error,
  onRetry,
  title,
}: {
  error: Error;
  onRetry?: () => void;
  title?: string;
}) {
  const status = error instanceof ApiError ? error.status : undefined;
  return (
    <Card className="animate-fade-in border-down/30 p-5">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-lg bg-down/10 p-2 text-down">
          <AlertTriangle size={18} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-ink">
              {title || 'Request failed'}
            </h3>
            {status !== undefined && (
              <span className="rounded-md border border-down/30 bg-down/10 px-1.5 py-0.5 font-mono text-2xs text-down">
                HTTP {status}
              </span>
            )}
          </div>
          <p className="mt-1 break-words font-mono text-xs text-muted">
            {error.message}
          </p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="btn-ghost mt-3 !py-1.5 text-xs"
            >
              <RefreshCw size={13} /> Retry
            </button>
          )}
        </div>
      </div>
    </Card>
  );
}

export function EmptyState({
  icon,
  title,
  hint,
  action,
}: {
  icon?: ReactNode;
  title: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-line bg-panel/40 px-6 py-14 text-center animate-fade-in">
      <div className="rounded-xl bg-panel-2 p-3 text-faint">
        {icon || <Inbox size={22} />}
      </div>
      <div>
        <p className="text-sm font-medium text-ink">{title}</p>
        {hint && <p className="mt-1 max-w-sm text-xs text-muted">{hint}</p>}
      </div>
      {action}
    </div>
  );
}

export function LoadingRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-11 animate-pulse rounded-lg bg-panel-2"
          style={{ animationDelay: `${i * 70}ms` }}
        />
      ))}
    </div>
  );
}

export function CenterSpinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-14 text-sm text-muted">
      <Spinner className="h-4 w-4" />
      {label || 'Loading…'}
    </div>
  );
}

/**
 * Standard loading / error / empty / content switch used by every data view.
 */
export function StateView<T>({
  loading,
  error,
  data,
  onRetry,
  isEmpty,
  empty,
  skeleton,
  children,
}: {
  loading: boolean;
  error: Error | null;
  data: T | null;
  onRetry?: () => void;
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  skeleton?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (error && !data) return <ErrorCard error={error} onRetry={onRetry} />;
  if (loading && !data) return <>{skeleton || <CenterSpinner />}</>;
  if (!data) return <>{skeleton || <CenterSpinner />}</>;
  if (isEmpty && isEmpty(data)) return <>{empty || <EmptyState title="Nothing here yet" />}</>;
  return <>{children(data)}</>;
}

export const AuthIcon = KeyRound;
