import { useState, type ReactNode } from 'react';
import { Check, Copy, Loader2 } from 'lucide-react';
import { classNames } from '../lib/format';

export function Card({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={classNames('card', className)}>{children}</div>;
}

export function SectionTitle({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <h2 className="text-sm font-semibold tracking-tight text-ink">{title}</h2>
        {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={classNames('animate-spin', className)} aria-hidden />;
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={classNames(
        'relative overflow-hidden rounded-md bg-panel-2',
        className,
      )}
    >
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-black/5 to-transparent dark:via-white/5" />
    </div>
  );
}

type BadgeTone = 'neutral' | 'ok' | 'warn' | 'down' | 'brand' | 'info';

const badgeTones: Record<BadgeTone, string> = {
  neutral: 'border-line bg-panel-2 text-muted',
  ok: 'border-ok/30 bg-ok/10 text-ok',
  warn: 'border-warn/30 bg-warn/10 text-warn',
  down: 'border-down/30 bg-down/10 text-down',
  brand: 'border-brand/30 bg-brand/10 text-brand',
  info: 'border-info/30 bg-info/10 text-info',
};

export function Badge({
  children,
  tone = 'neutral',
  mono,
}: {
  children: ReactNode;
  tone?: BadgeTone;
  mono?: boolean;
}) {
  return (
    <span
      className={classNames(
        'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-2xs font-medium',
        mono && 'font-mono',
        badgeTones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function CopyButton({ value, label }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard blocked; no-op */
    }
  };
  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex items-center gap-1 rounded-md border border-line bg-panel px-1.5 py-1 text-2xs text-muted transition-colors hover:text-ink"
      aria-label={label || 'Copy to clipboard'}
    >
      {copied ? (
        <Check size={12} className="text-ok" />
      ) : (
        <Copy size={12} />
      )}
      {label}
    </button>
  );
}

export function Mono({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <span className={classNames('font-mono text-ink', className)}>{children}</span>
  );
}
