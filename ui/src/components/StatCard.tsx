import type { ReactNode } from 'react';
import { Card } from './primitives';
import { Sparkline } from './Sparkline';
import { classNames } from '../lib/format';

export function StatCard({
  label,
  value,
  unit,
  icon,
  series,
  color,
  hint,
  loading,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  icon?: ReactNode;
  series?: number[];
  color?: string;
  hint?: string;
  loading?: boolean;
}) {
  return (
    <Card className="group relative overflow-hidden p-4 transition-colors hover:border-line-strong">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium text-muted">
          {icon && <span className="text-faint">{icon}</span>}
          {label}
        </span>
      </div>
      <div className="mt-2 flex items-end justify-between gap-2">
        <div className="min-w-0">
          <span
            className={classNames(
              'font-mono text-2xl font-semibold tracking-tight text-ink tabular-nums',
              loading && 'opacity-40',
            )}
          >
            {value}
          </span>
          {unit && <span className="ml-1 text-xs text-faint">{unit}</span>}
        </div>
        {series && series.length > 1 && (
          <Sparkline values={series} color={color} className="shrink-0 opacity-90" />
        )}
      </div>
      {hint && <p className="mt-1 truncate text-2xs text-faint">{hint}</p>}
    </Card>
  );
}
