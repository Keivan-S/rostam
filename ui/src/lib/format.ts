// Small formatting helpers for metric numbers, bytes, durations, and latency.

export function formatNumber(n: number | undefined | null): string {
  if (n === undefined || n === null || Number.isNaN(n)) return '—';
  if (!Number.isFinite(n)) return '∞';
  if (Math.abs(n) >= 1_000_000_000)
    return (n / 1_000_000_000).toFixed(n % 1_000_000_000 === 0 ? 0 : 1) + 'B';
  if (Math.abs(n) >= 1_000_000)
    return (n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1) + 'M';
  if (Math.abs(n) >= 10_000) return (n / 1_000).toFixed(1) + 'k';
  return n.toLocaleString();
}

export function formatInt(n: number | undefined | null): string {
  if (n === undefined || n === null || Number.isNaN(n)) return '—';
  return Math.round(n).toLocaleString();
}

export function formatBytes(bytes: number | undefined | null): string {
  if (bytes === undefined || bytes === null || Number.isNaN(bytes)) return '—';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const i = Math.min(
    units.length - 1,
    Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024)),
  );
  const v = bytes / Math.pow(1024, i);
  return `${v.toFixed(i === 0 ? 0 : v >= 100 ? 0 : 1)} ${units[i]}`;
}

export function formatPercent(frac: number | undefined | null, digits = 1): string {
  if (frac === undefined || frac === null || Number.isNaN(frac)) return '—';
  return `${(frac * 100).toFixed(digits)}%`;
}

/** Format a latency given in SECONDS as a human ms/µs/s string. */
export function formatLatency(seconds: number | undefined | null): string {
  if (seconds === undefined || seconds === null || Number.isNaN(seconds)) return '—';
  if (seconds === 0) return '0';
  const ms = seconds * 1000;
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 100) return `${ms.toFixed(2)} ms`;
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${seconds.toFixed(2)} s`;
}

export function formatTtl(ms: number | undefined | null): string {
  if (ms === undefined || ms === null) return 'no expiry';
  if (ms < 0) return 'no expiry';
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

export function classNames(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}
