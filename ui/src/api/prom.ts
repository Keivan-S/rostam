// A small, defensive Prometheus text-exposition parser.
//
// It parses `# HELP` / `# TYPE` lines and sample lines of the form
//   metric_name{label="v",other="w"} 12.3
// It never throws on malformed input: unparseable lines are skipped. Metric
// names carry a `rostam_vector_` prefix in this server, so callers match by
// suffix (see suffixMatch) to stay robust to prefix changes.

export interface Sample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

export interface ParsedMetrics {
  samples: Sample[];
  /** metric name -> HELP/TYPE metadata, best-effort. */
  meta: Record<string, { help?: string; type?: string }>;
}

function parseLabels(raw: string): Record<string, string> {
  const labels: Record<string, string> = {};
  // Match key="value" pairs; value may contain escaped quotes/backslashes.
  const re = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(raw)) !== null) {
    labels[m[1]] = m[2]
      .replace(/\\"/g, '"')
      .replace(/\\n/g, '\n')
      .replace(/\\\\/g, '\\');
  }
  return labels;
}

export function parsePrometheus(text: string): ParsedMetrics {
  const samples: Sample[] = [];
  const meta: Record<string, { help?: string; type?: string }> = {};

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (line === '') continue;

    if (line.startsWith('#')) {
      // "# HELP name text" or "# TYPE name type"
      const parts = line.split(/\s+/);
      const kind = parts[1];
      const name = parts[2];
      if (!name) continue;
      if (kind === 'HELP') {
        meta[name] = { ...meta[name], help: parts.slice(3).join(' ') };
      } else if (kind === 'TYPE') {
        meta[name] = { ...meta[name], type: parts[3] };
      }
      continue;
    }

    // sample: name{labels} value  OR  name value
    const braceIdx = line.indexOf('{');
    let name: string;
    let labels: Record<string, string> = {};
    let rest: string;

    if (braceIdx >= 0) {
      const closeIdx = line.lastIndexOf('}');
      if (closeIdx < braceIdx) continue;
      name = line.slice(0, braceIdx).trim();
      labels = parseLabels(line.slice(braceIdx + 1, closeIdx));
      rest = line.slice(closeIdx + 1).trim();
    } else {
      const sp = line.indexOf(' ');
      if (sp < 0) continue;
      name = line.slice(0, sp).trim();
      rest = line.slice(sp + 1).trim();
    }

    if (name === '') continue;
    // value is the first token after labels (ignore an optional timestamp).
    const valTok = rest.split(/\s+/)[0];
    const value = valTok === '+Inf' ? Infinity : valTok === '-Inf' ? -Infinity : Number(valTok);
    if (Number.isNaN(value)) continue;

    samples.push({ name, labels, value });
  }

  return { samples, meta };
}

/** Does a metric name end with the given suffix (prefix-robust matching)? */
export function suffixMatch(name: string, suffix: string): boolean {
  return name === suffix || name.endsWith('_' + suffix) || name.endsWith(suffix);
}

/**
 * Estimate a quantile from cumulative histogram buckets.
 * `buckets` is a list of { le, count } where count is the CUMULATIVE count at
 * that upper bound (Prometheus convention). Returns the interpolated bound in
 * the histogram's native unit (seconds here). Returns null when there is no data.
 */
export function histogramQuantile(
  buckets: { le: number; count: number }[],
  q: number,
): number | null {
  if (buckets.length === 0) return null;
  const sorted = [...buckets].sort((a, b) => a.le - b.le);
  const total = sorted[sorted.length - 1].count;
  if (total <= 0) return null;
  const target = q * total;

  let prevCount = 0;
  let prevLe = 0;
  for (const b of sorted) {
    if (b.count >= target) {
      if (!Number.isFinite(b.le)) {
        // Target falls in the +Inf bucket; best estimate is the last finite bound.
        return prevLe > 0 ? prevLe : null;
      }
      const bucketCount = b.count - prevCount;
      if (bucketCount <= 0) return b.le;
      // Linear interpolation within the bucket [prevLe, le].
      const frac = (target - prevCount) / bucketCount;
      return prevLe + (b.le - prevLe) * frac;
    }
    prevCount = b.count;
    prevLe = Number.isFinite(b.le) ? b.le : prevLe;
  }
  return prevLe > 0 ? prevLe : null;
}
