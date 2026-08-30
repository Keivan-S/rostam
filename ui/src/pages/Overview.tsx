import {
  Activity,
  Boxes,
  Crown,
  Database,
  Gauge,
  HardDrive,
  Layers,
  Timer,
  Zap,
} from 'lucide-react';
import { PageHeader } from '../components/Layout';
import { StatCard } from '../components/StatCard';
import { Card, SectionTitle, Badge } from '../components/primitives';
import { EmptyState, ErrorCard } from '../components/StateView';
import { HealthDot } from '../components/HealthDot';
import { useMetrics, type HistoryPoint } from '../hooks/useMetrics';
import { useHealth } from '../hooks/useHealth';
import { useAsync } from '../hooks/useAsync';
import { getReplication, getTopology } from '../api/endpoints';
import { ApiError } from '../api/client';
import {
  formatBytes,
  formatInt,
  formatLatency,
  formatNumber,
  formatPercent,
} from '../lib/format';
import type { ReplicationResponse, TopologyResponse } from '../api/types';

const pick = (h: HistoryPoint[], key: keyof HistoryPoint) =>
  h.map((p) => Number(p[key]) || 0);

export function Overview() {
  const { snapshot, history, error } = useMetrics();
  const { state, detail } = useHealth();
  const agg = snapshot?.aggregate;

  const topo = useAsync<TopologyResponse>((s) => getTopology(s), []);
  const repl = useAsync<ReplicationResponse>((s) => getReplication(s), []);

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Overview"
        description="Live cluster health, throughput, and shard placement."
        actions={
          <div className="flex items-center gap-2 rounded-lg border border-line bg-panel px-3 py-2">
            <HealthDot state={state} detail={detail} />
          </div>
        }
      />

      {error && !snapshot && (
        <div className="mb-5">
          <ErrorCard error={error} title="Could not load /metrics" />
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard
          label="Points"
          icon={<Layers size={13} />}
          value={formatNumber(agg?.totalPoints)}
          series={pick(history, 'points')}
          loading={!snapshot}
          hint={agg ? `${formatInt(agg.totalTombstoned)} tombstoned` : undefined}
        />
        <StatCard
          label="Collections"
          icon={<Boxes size={13} />}
          value={formatInt(agg?.collectionCount)}
          loading={!snapshot}
        />
        <StatCard
          label="Search QPS"
          icon={<Zap size={13} />}
          value={agg?.searchQps == null ? '—' : formatNumber(agg.searchQps)}
          series={pick(history, 'searchQps')}
          color="rgb(var(--c-info))"
          loading={!snapshot}
          hint={agg?.insertQps != null ? `${formatNumber(agg.insertQps)} insert/s` : undefined}
        />
        <StatCard
          label="p99 search"
          icon={<Timer size={13} />}
          value={formatLatency(agg?.p99SearchSec)}
          series={pick(history, 'p99SearchMs')}
          color="rgb(var(--c-warn))"
          loading={!snapshot}
        />
        <StatCard
          label="Cache hit rate"
          icon={<Gauge size={13} />}
          value={agg?.cacheHitRate == null ? 'n/a' : formatPercent(agg.cacheHitRate)}
          series={agg?.cacheHitRate == null ? undefined : pick(history, 'hitRate')}
          color="rgb(var(--c-ok))"
          loading={!snapshot}
          hint={agg?.cacheHitRate == null ? 'not exposed' : undefined}
        />
        <StatCard
          label="Memory used"
          icon={<HardDrive size={13} />}
          value={agg?.memoryUsedBytes == null ? 'n/a' : formatBytes(agg.memoryUsedBytes)}
          series={agg?.memoryUsedBytes == null ? undefined : pick(history, 'memoryMiB')}
          loading={!snapshot}
          hint={
            agg?.memoryAllocatedBytes != null
              ? `${formatBytes(agg.memoryAllocatedBytes)} allocated`
              : agg?.memoryUsedBytes == null
                ? 'not exposed'
                : undefined
          }
        />
      </div>

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <PlacementCard topo={topo.data} error={topo.error} loading={topo.loading} onRetry={topo.reload} />
        </div>
        <NodesCard topo={topo.data} />
      </div>

      <div className="mt-5">
        <ReplicationCard repl={repl.data} error={repl.error} loading={repl.loading} onRetry={repl.reload} />
      </div>
    </div>
  );
}

// --- Shard placement grid ----------------------------------------------------

function PlacementCard({
  topo,
  error,
  loading,
  onRetry,
}: {
  topo: TopologyResponse | null;
  error: Error | null;
  loading: boolean;
  onRetry: () => void;
}) {
  const unavailable = error instanceof ApiError && error.isUnavailable;
  return (
    <Card className="p-4">
      <SectionTitle
        title="Shard placement"
        subtitle="Each row is a shard; the leader replica is highlighted."
      />
      <div className="mt-4">
        {error ? (
          unavailable ? (
            <EmptyState
              icon={<Database size={20} />}
              title="Topology not available"
              hint="This node is running single-node, or the topology endpoint is not enabled."
            />
          ) : loading ? null : (
            <ErrorCard error={error} onRetry={onRetry} />
          )
        ) : !topo ? (
          <div className="h-24 animate-pulse rounded-lg bg-panel-2" />
        ) : topo.num_shards === 0 ? (
          <EmptyState title="No shards" hint="Single-node deployment." />
        ) : (
          <div className="space-y-1.5">
            {Array.from({ length: topo.num_shards }).map((_, shard) => {
              const nodes = topo.placement?.[shard] || [];
              const leader = topo.leaders?.[shard];
              return (
                <div key={shard} className="flex items-center gap-2">
                  <span className="w-16 shrink-0 font-mono text-2xs text-faint">
                    shard {shard}
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {nodes.length === 0 && (
                      <span className="text-2xs text-faint">unplaced</span>
                    )}
                    {nodes.map((nodeId) => {
                      const isLeader = isLeaderNode(topo, nodeId, leader);
                      return (
                        <span
                          key={nodeId}
                          className={
                            'inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-2xs ' +
                            (isLeader
                              ? 'border-brand/40 bg-brand/10 text-brand'
                              : 'border-line bg-panel-2 text-muted')
                          }
                        >
                          {isLeader && <Crown size={10} />}
                          {nodeId}
                        </span>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
}

// The leaders array holds a server_addr; map it back to a node_id for highlight.
function isLeaderNode(
  topo: TopologyResponse,
  nodeId: string,
  leaderAddr: string | undefined,
): boolean {
  if (!leaderAddr) return false;
  const member = topo.members?.find((m) => m.node_id === nodeId);
  return member?.server_addr === leaderAddr || nodeId === leaderAddr;
}

// --- Node list ---------------------------------------------------------------

function NodesCard({ topo }: { topo: TopologyResponse | null }) {
  const members = topo?.members || [];
  return (
    <Card className="p-4">
      <SectionTitle title="Nodes" subtitle={`${members.length} member${members.length === 1 ? '' : 's'}`} />
      <div className="mt-4 space-y-2">
        {!topo ? (
          <div className="h-20 animate-pulse rounded-lg bg-panel-2" />
        ) : members.length === 0 ? (
          <EmptyState title="No members" hint="Single-node deployment." />
        ) : (
          members.map((m) => (
            <div
              key={m.node_id}
              className="flex items-center justify-between rounded-lg border border-line bg-panel-2 px-3 py-2"
            >
              <div className="flex items-center gap-2">
                <Activity size={13} className="text-ok" />
                <span className="font-mono text-xs text-ink">{m.node_id}</span>
              </div>
              <span className="font-mono text-2xs text-faint">{m.server_addr}</span>
            </div>
          ))
        )}
      </div>
    </Card>
  );
}

// --- Replication -------------------------------------------------------------

function ReplicationCard({
  repl,
  error,
  loading,
  onRetry,
}: {
  repl: ReplicationResponse | null;
  error: Error | null;
  loading: boolean;
  onRetry: () => void;
}) {
  const shards = repl?.shards || [];
  return (
    <Card className="p-4">
      <SectionTitle
        title="Replication health"
        subtitle="Primary / ISR vs min-ISR and per-backup lag for each hosted shard."
      />
      <div className="mt-4">
        {error ? (
          loading ? null : <ErrorCard error={error} onRetry={onRetry} />
        ) : !repl ? (
          <div className="h-16 animate-pulse rounded-lg bg-panel-2" />
        ) : shards.length === 0 ? (
          <EmptyState
            title="No replicated shards"
            hint="Single-node or Raft-replicated mode reports no primary-backup shards here."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-2xs uppercase tracking-wide text-faint">
                <tr className="border-b border-line">
                  <th className="py-2 pr-4 font-medium">Shard</th>
                  <th className="py-2 pr-4 font-medium">Mode</th>
                  <th className="py-2 pr-4 font-medium">Primary</th>
                  <th className="py-2 pr-4 font-medium">ISR</th>
                  <th className="py-2 pr-4 font-medium">Backups / lag</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {shards.map((s, i) => {
                  const healthy = s.isr != null && s.min_isr != null && s.isr >= s.min_isr;
                  return (
                    <tr key={i} className="border-b border-line/60">
                      <td className="py-2 pr-4 text-ink">{s.shard ?? i}</td>
                      <td className="py-2 pr-4 text-muted">{s.mode ?? '—'}</td>
                      <td className="py-2 pr-4 text-muted">{s.primary ?? '—'}</td>
                      <td className="py-2 pr-4">
                        <Badge tone={healthy ? 'ok' : 'warn'} mono>
                          {s.isr ?? '?'} / {s.min_isr ?? '?'}
                        </Badge>
                      </td>
                      <td className="py-2 pr-4 text-muted">
                        {(s.backups || []).length === 0
                          ? '—'
                          : (s.backups || [])
                              .map((b) => `${b.node_id ?? '?'}:${b.lag ?? 0}`)
                              .join('  ')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Card>
  );
}
