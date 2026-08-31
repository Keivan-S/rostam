import { useCallback, useEffect, useRef, useState } from 'react';
import { getMetricsText } from '../api/endpoints';
import { ApiError } from '../api/client';
import {
  histogramQuantile,
  parsePrometheus,
  suffixMatch,
  type Sample,
} from '../api/prom';
import { useApiKey } from '../context/ApiKeyContext';
import { useSettings } from '../context/SettingsContext';

export interface CollectionMetrics {
  name: string;
  size: number;
  tombstoned: number;
  searchOps: number;
  insertOps: number;
  expired: number;
  searchQps: number | null;
  insertQps: number | null;
  p99SearchSec: number | null;
  p99InsertSec: number | null;
}

export interface AggregateMetrics {
  totalPoints: number;
  totalTombstoned: number;
  collectionCount: number;
  searchQps: number | null;
  insertQps: number | null;
  p99SearchSec: number | null;
  cacheHitRate: number | null;
  memoryUsedBytes: number | null;
  memoryAllocatedBytes: number | null;
}

export interface MetricsSnapshot {
  collections: CollectionMetrics[];
  aggregate: AggregateMetrics;
  at: number;
}

export interface HistoryPoint {
  t: number;
  points: number;
  searchQps: number;
  p99SearchMs: number;
  hitRate: number;
  memoryMiB: number;
}

const HISTORY_LEN = 60;

interface RawByCollection {
  size: number;
  tombstoned: number;
  searchOps: number;
  insertOps: number;
  expired: number;
  searchBuckets: { le: number; count: number }[];
  insertBuckets: { le: number; count: number }[];
}

function leValue(v: string): number {
  return v === '+Inf' ? Infinity : Number(v);
}

// Group samples into per-collection raw counters + histogram buckets, plus a
// best-effort cache view (the backend may or may not expose cache metrics here).
function reduceSamples(samples: Sample[]) {
  const byCol = new Map<string, RawByCollection>();
  const ensure = (name: string): RawByCollection => {
    let r = byCol.get(name);
    if (!r) {
      r = {
        size: 0,
        tombstoned: 0,
        searchOps: 0,
        insertOps: 0,
        expired: 0,
        searchBuckets: [],
        insertBuckets: [],
      };
      byCol.set(name, r);
    }
    return r;
  };

  let cacheHits: number | null = null;
  let cacheGets: number | null = null;
  let bytesUsed: number | null = null;
  let bytesAllocated: number | null = null;

  for (const s of samples) {
    const col = s.labels['collection'];
    const isCache = s.name.includes('cache');

    if (col !== undefined) {
      const r = ensure(col);
      if (suffixMatch(s.name, 'size')) r.size = s.value;
      else if (suffixMatch(s.name, 'tombstoned')) r.tombstoned = s.value;
      else if (suffixMatch(s.name, 'search_ops_total')) r.searchOps = s.value;
      else if (suffixMatch(s.name, 'insert_ops_total')) r.insertOps = s.value;
      else if (suffixMatch(s.name, 'expired_total')) r.expired = s.value;
      else if (s.name.includes('search_latency_seconds_bucket') && s.labels['le'] !== undefined)
        r.searchBuckets.push({ le: leValue(s.labels['le']), count: s.value });
      else if (s.name.includes('insert_latency_seconds_bucket') && s.labels['le'] !== undefined)
        r.insertBuckets.push({ le: leValue(s.labels['le']), count: s.value });
    }

    // Best-effort cache metrics (may be absent in the current server).
    if (isCache) {
      if (suffixMatch(s.name, 'hits') || suffixMatch(s.name, 'hits_total'))
        cacheHits = (cacheHits ?? 0) + s.value;
      else if (suffixMatch(s.name, 'gets') || suffixMatch(s.name, 'gets_total'))
        cacheGets = (cacheGets ?? 0) + s.value;
      else if (suffixMatch(s.name, 'bytes_used'))
        bytesUsed = (bytesUsed ?? 0) + s.value;
      else if (suffixMatch(s.name, 'bytes_allocated'))
        bytesAllocated = (bytesAllocated ?? 0) + s.value;
    }
  }

  const cacheHitRate =
    cacheHits !== null && cacheGets !== null && cacheGets > 0
      ? cacheHits / cacheGets
      : null;

  return { byCol, cacheHitRate, bytesUsed, bytesAllocated };
}

export interface UseMetricsResult {
  snapshot: MetricsSnapshot | null;
  history: HistoryPoint[];
  error: Error | null;
  loading: boolean;
  paused: boolean;
  reload: () => void;
}

/**
 * Poll /metrics, diff against the previous sample to derive rates (QPS), and
 * expose parsed per-collection + aggregate values plus a rolling history for
 * sparklines. Polling pauses while the tab is hidden.
 */
export function useMetrics(): UseMetricsResult {
  const { pollMs } = useSettings();
  const { reportUnauthorized } = useApiKey();
  const [snapshot, setSnapshot] = useState<MetricsSnapshot | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [paused, setPaused] = useState(
    typeof document !== 'undefined' && document.hidden,
  );

  const prevRef = useRef<{
    at: number;
    byCol: Map<string, RawByCollection>;
  } | null>(null);
  const [nonce, setNonce] = useState(0);
  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    const onVis = () => setPaused(document.hidden);
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);

  useEffect(() => {
    if (paused) return;
    let active = true;
    let timer: number | undefined;
    // Hoisted so cleanup can abort an in-flight request, and reused as the
    // re-entrancy guard: tick() is self-scheduling (setTimeout, not
    // setInterval) so a slow fetch can never overlap the next one — the next
    // tick is scheduled only once the current one finishes.
    let ctrl: AbortController | null = null;

    const tick = async () => {
      ctrl = new AbortController();
      try {
        const text = await getMetricsText(ctrl.signal);
        if (!active) return;
        const { samples } = parsePrometheus(text);
        const { byCol, cacheHitRate, bytesUsed, bytesAllocated } =
          reduceSamples(samples);
        const now = Date.now();
        const prev = prevRef.current;
        const dt = prev ? (now - prev.at) / 1000 : 0;

        const collections: CollectionMetrics[] = [];
        let totalPoints = 0;
        let totalTombstoned = 0;
        let aggSearchQps: number | null = prev ? 0 : null;
        let aggInsertQps: number | null = prev ? 0 : null;

        for (const [name, r] of byCol) {
          totalPoints += r.size;
          totalTombstoned += r.tombstoned;
          let searchQps: number | null = null;
          let insertQps: number | null = null;
          if (prev && dt > 0) {
            const p = prev.byCol.get(name);
            if (p) {
              searchQps = Math.max(0, (r.searchOps - p.searchOps) / dt);
              insertQps = Math.max(0, (r.insertOps - p.insertOps) / dt);
              if (aggSearchQps !== null) aggSearchQps += searchQps;
              if (aggInsertQps !== null) aggInsertQps += insertQps;
            }
          }
          collections.push({
            name,
            size: r.size,
            tombstoned: r.tombstoned,
            searchOps: r.searchOps,
            insertOps: r.insertOps,
            expired: r.expired,
            searchQps,
            insertQps,
            p99SearchSec: histogramQuantile(r.searchBuckets, 0.99),
            p99InsertSec: histogramQuantile(r.insertBuckets, 0.99),
          });
        }
        collections.sort((a, b) => b.size - a.size || a.name.localeCompare(b.name));

        // Aggregate p99: fold every collection's search buckets together.
        const allSearchBuckets = new Map<number, number>();
        for (const r of byCol.values())
          for (const b of r.searchBuckets)
            allSearchBuckets.set(b.le, (allSearchBuckets.get(b.le) ?? 0) + b.count);
        const aggP99 = histogramQuantile(
          Array.from(allSearchBuckets, ([le, count]) => ({ le, count })),
          0.99,
        );

        const aggregate: AggregateMetrics = {
          totalPoints,
          totalTombstoned,
          collectionCount: byCol.size,
          searchQps: aggSearchQps,
          insertQps: aggInsertQps,
          p99SearchSec: aggP99,
          cacheHitRate,
          memoryUsedBytes: bytesUsed,
          memoryAllocatedBytes: bytesAllocated,
        };

        setSnapshot({ collections, aggregate, at: now });
        setError(null);
        setLoading(false);

        if (prev) {
          setHistory((h) => {
            const next = [
              ...h,
              {
                t: now,
                points: totalPoints,
                searchQps: aggSearchQps ?? 0,
                p99SearchMs: (aggP99 ?? 0) * 1000,
                hitRate: cacheHitRate ?? 0,
                memoryMiB: (bytesUsed ?? 0) / (1024 * 1024),
              },
            ];
            return next.length > HISTORY_LEN
              ? next.slice(next.length - HISTORY_LEN)
              : next;
          });
        }
        prevRef.current = { at: now, byCol };
      } catch (e) {
        if (!active || (e instanceof DOMException && e.name === 'AbortError')) return;
        if (e instanceof ApiError && e.status === 401) reportUnauthorized();
        setError(e instanceof Error ? e : new Error(String(e)));
        setLoading(false);
      } finally {
        ctrl = null;
        if (active) timer = window.setTimeout(tick, pollMs);
      }
    };

    tick();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
      ctrl?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pollMs, paused, nonce]);

  return { snapshot, history, error, loading, paused, reload };
}
