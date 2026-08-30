import { useMemo, useState } from 'react';
import { Boxes, ChevronRight, Plus, RefreshCw } from 'lucide-react';
import { PageHeader } from '../components/Layout';
import { Card } from '../components/primitives';
import { EmptyState, ErrorCard, LoadingRows } from '../components/StateView';
import { CreateCollectionForm } from '../components/CreateCollectionForm';
import { CollectionDetail } from '../components/CollectionDetail';
import { useAsync } from '../hooks/useAsync';
import { useMetrics, type CollectionMetrics } from '../hooks/useMetrics';
import { listCollections } from '../api/endpoints';
import { ApiError } from '../api/client';
import { formatInt, formatLatency, formatNumber } from '../lib/format';
import type { CollectionListResponse } from '../api/types';

interface Row {
  name: string;
  metrics?: CollectionMetrics;
}

export function Collections() {
  const list = useAsync<CollectionListResponse>((s) => listCollections(s), []);
  const { snapshot } = useMetrics();
  const [createOpen, setCreateOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const metricsByName = useMemo(() => {
    const map = new Map<string, CollectionMetrics>();
    snapshot?.collections.forEach((c) => map.set(c.name, c));
    return map;
  }, [snapshot]);

  // Prefer the explicit collections list; fall back to names seen in /metrics
  // when that endpoint is unavailable on this build.
  const listUnavailable = list.error instanceof ApiError && list.error.isUnavailable;
  const rows: Row[] = useMemo(() => {
    const names = new Set<string>();
    list.data?.collections?.forEach((c) => names.add(c.name));
    if (listUnavailable || (!list.data && snapshot))
      snapshot?.collections.forEach((c) => names.add(c.name));
    return Array.from(names)
      .sort((a, b) => a.localeCompare(b))
      .map((name) => ({ name, metrics: metricsByName.get(name) }));
  }, [list.data, listUnavailable, snapshot, metricsByName]);

  const refresh = () => list.reload();

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Collections"
        description="Vector collections on this server, with live throughput."
        actions={
          <>
            <button type="button" className="btn-ghost" onClick={refresh} aria-label="Refresh">
              <RefreshCw size={14} />
            </button>
            <button type="button" className="btn-primary" onClick={() => setCreateOpen(true)}>
              <Plus size={15} /> Create collection
            </button>
          </>
        }
      />

      {listUnavailable && (
        <p className="mb-4 rounded-lg border border-line bg-panel-2 px-3 py-2 text-xs text-muted">
          The collection list endpoint is not available on this build — showing
          collections discovered from <span className="font-mono">/metrics</span> instead.
        </p>
      )}

      {list.error && !listUnavailable && !list.data ? (
        <ErrorCard error={list.error} onRetry={refresh} title="Could not list collections" />
      ) : list.loading && !list.data && rows.length === 0 ? (
        <Card className="p-4">
          <LoadingRows rows={5} />
        </Card>
      ) : rows.length === 0 ? (
        <EmptyState
          icon={<Boxes size={22} />}
          title="No collections yet"
          hint="Create your first vector collection to start indexing embeddings."
          action={
            <button type="button" className="btn-primary" onClick={() => setCreateOpen(true)}>
              <Plus size={15} /> Create collection
            </button>
          }
        />
      ) : (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-line text-2xs uppercase tracking-wide text-faint">
                  <th className="px-4 py-2.5 font-medium">Collection</th>
                  <th className="px-4 py-2.5 text-right font-medium">Points</th>
                  <th className="px-4 py-2.5 text-right font-medium">Tombstoned</th>
                  <th className="px-4 py-2.5 text-right font-medium">Search QPS</th>
                  <th className="px-4 py-2.5 text-right font-medium">p99 search</th>
                  <th className="w-8 px-4 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.name}
                    onClick={() => setSelected(r.name)}
                    className="group cursor-pointer border-b border-line/50 transition-colors last:border-0 hover:bg-panel-2/50"
                  >
                    <td className="px-4 py-3">
                      <span className="font-mono text-ink">{r.name}</span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-ink tabular-nums">
                      {formatNumber(r.metrics?.size)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-muted tabular-nums">
                      {formatInt(r.metrics?.tombstoned)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-muted tabular-nums">
                      {r.metrics?.searchQps == null ? '—' : formatNumber(r.metrics.searchQps)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-muted tabular-nums">
                      {formatLatency(r.metrics?.p99SearchSec)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <ChevronRight
                        size={15}
                        className="text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-muted"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <CreateCollectionForm
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => refresh()}
      />

      {selected && (
        <CollectionDetail
          name={selected}
          metrics={metricsByName.get(selected)}
          onClose={() => setSelected(null)}
          onChanged={refresh}
        />
      )}
    </div>
  );
}
