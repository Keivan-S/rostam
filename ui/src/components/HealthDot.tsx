import { classNames } from '../lib/format';
import type { HealthState } from '../api/types';

const meta: Record<HealthState, { color: string; label: string; pulse: boolean }> = {
  ready: { color: 'bg-ok', label: 'Ready', pulse: false },
  notready: { color: 'bg-warn', label: 'Not ready', pulse: true },
  down: { color: 'bg-down', label: 'Unreachable', pulse: true },
  unknown: { color: 'bg-faint', label: 'Checking…', pulse: true },
};

export function HealthDot({
  state,
  detail,
  showLabel = true,
}: {
  state: HealthState;
  detail?: string;
  showLabel?: boolean;
}) {
  const m = meta[state];
  return (
    <span
      className="inline-flex items-center gap-2"
      title={detail || m.label}
      role="status"
      aria-label={`Cluster status: ${m.label}`}
    >
      <span className="relative flex h-2.5 w-2.5">
        {m.pulse && (
          <span
            className={classNames(
              'absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping',
              m.color,
            )}
          />
        )}
        <span
          className={classNames(
            'relative inline-flex h-2.5 w-2.5 rounded-full',
            m.color,
          )}
        />
      </span>
      {showLabel && (
        <span className="text-xs font-medium text-muted">{m.label}</span>
      )}
    </span>
  );
}
