import { useState } from 'react';
import { Ruler, Trash2, Split, Fingerprint } from 'lucide-react';
import { Drawer } from './Drawer';
import { Modal, TypedConfirmModal } from './Modal';
import { Badge, CopyButton } from './primitives';
import { ErrorCard } from './StateView';
import { useAsync } from '../hooks/useAsync';
import { getCollection, dropCollection, reshardCollection } from '../api/endpoints';
import { ApiError } from '../api/client';
import { formatInt, formatLatency, formatNumber } from '../lib/format';
import type { CollectionMetrics } from '../hooks/useMetrics';
import type { CollectionConfig } from '../api/types';

// Config keys surfaced prominently; the rest render in the generic table.
const PROMINENT = new Set(['dim', 'metric']);

export function CollectionDetail({
  name,
  metrics,
  onClose,
  onChanged,
}: {
  name: string;
  metrics?: CollectionMetrics;
  onClose: () => void;
  onChanged: () => void;
}) {
  const cfg = useAsync<CollectionConfig>((s) => getCollection(name, s), [name]);
  const [dropOpen, setDropOpen] = useState(false);
  const [reshardOpen, setReshardOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const doDrop = async () => {
    setBusy(true);
    setActionError(null);
    try {
      await dropCollection(name);
      setBusy(false);
      setDropOpen(false);
      onChanged();
      onClose();
    } catch (e) {
      setBusy(false);
      setActionError(e instanceof Error ? e.message : 'Drop failed.');
    }
  };

  const cfgUnavailable = cfg.error instanceof ApiError && cfg.error.isUnavailable;

  return (
    <Drawer
      open
      onClose={onClose}
      title={<span className="font-mono">{name}</span>}
      subtitle="Collection configuration &amp; live metrics"
      footer={
        <>
          <button type="button" className="btn-ghost" onClick={() => setReshardOpen(true)}>
            <Split size={14} /> Reshard
          </button>
          <button type="button" className="btn-danger" onClick={() => setDropOpen(true)}>
            <Trash2 size={14} /> Drop
          </button>
        </>
      }
    >
      {/* Prominent config */}
      <div className="grid grid-cols-2 gap-3">
        <Prominent
          icon={<Ruler size={14} />}
          label="Dimension"
          value={cfg.data?.dim != null ? String(cfg.data.dim) : '—'}
        />
        <Prominent
          icon={<Fingerprint size={14} />}
          label="Metric"
          value={cfg.data?.metric ?? '—'}
        />
      </div>

      {/* Live metrics */}
      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label="Points" value={formatNumber(metrics?.size)} />
        <MetricTile label="Tombstoned" value={formatInt(metrics?.tombstoned)} />
        <MetricTile
          label="Search QPS"
          value={metrics?.searchQps == null ? '—' : formatNumber(metrics.searchQps)}
        />
        <MetricTile label="p99 search" value={formatLatency(metrics?.p99SearchSec)} />
      </div>

      {/* Full config */}
      <div className="mt-5">
        <div className="mb-2 text-xs font-medium text-muted">Configuration</div>
        {cfg.error ? (
          cfgUnavailable ? (
            <p className="rounded-lg border border-line bg-panel-2 px-3 py-2 text-xs text-muted">
              Config endpoint not available on this build.
            </p>
          ) : cfg.loading ? null : (
            <ErrorCard error={cfg.error} onRetry={cfg.reload} />
          )
        ) : !cfg.data ? (
          <div className="h-24 animate-pulse rounded-lg bg-panel-2" />
        ) : (
          <ConfigTable config={cfg.data} />
        )}
      </div>

      {actionError && (
        <p className="mt-4 rounded-lg border border-down/30 bg-down/10 px-3 py-2 font-mono text-xs text-down">
          {actionError}
        </p>
      )}

      <TypedConfirmModal
        open={dropOpen}
        onClose={() => setDropOpen(false)}
        onConfirm={doDrop}
        title="Drop collection"
        phrase={name}
        confirmLabel="Drop collection"
        busy={busy}
      >
        <p className="text-sm text-muted">
          This permanently deletes <span className="font-mono text-ink">{name}</span> and
          all of its points. This cannot be undone.
        </p>
      </TypedConfirmModal>

      <ReshardModal
        open={reshardOpen}
        name={name}
        onClose={() => setReshardOpen(false)}
        onDone={() => {
          setReshardOpen(false);
          onChanged();
        }}
      />
    </Drawer>
  );
}

function Prominent({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-line bg-panel-2 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-2xs text-faint">
        {icon}
        {label}
      </div>
      <div className="mt-1 font-mono text-lg font-semibold text-ink">{value}</div>
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line px-3 py-2">
      <div className="text-2xs text-faint">{label}</div>
      <div className="mt-0.5 font-mono text-sm font-medium text-ink">{value}</div>
    </div>
  );
}

function ConfigTable({ config }: { config: CollectionConfig }) {
  const entries = Object.entries(config).filter(([k]) => !PROMINENT.has(k));
  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <table className="w-full text-left text-xs">
        <tbody>
          {entries.map(([k, v], i) => (
            <tr key={k} className={i % 2 ? 'bg-panel-2/40' : ''}>
              <td className="w-1/2 py-1.5 pl-3 pr-2 font-mono text-muted">{k}</td>
              <td className="py-1.5 pr-3 font-mono text-ink">
                <div className="flex items-center justify-between gap-2">
                  <span className="break-all">{renderVal(v)}</span>
                  {typeof v !== 'object' && (
                    <CopyButton value={String(v)} />
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function renderVal(v: unknown): string {
  if (v === null) return 'null';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function ReshardModal({
  open,
  name,
  onClose,
  onDone,
}: {
  open: boolean;
  name: string;
  onClose: () => void;
  onDone: () => void;
}) {
  const [parts, setParts] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    const n = Number(parts);
    if (!Number.isInteger(n) || n <= 0) return setError('Enter a positive partition count.');
    setBusy(true);
    setError(null);
    try {
      await reshardCollection(name, n);
      setBusy(false);
      setParts('');
      onDone();
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : 'Reshard failed.');
    }
  };

  return (
    <Modal
      open={open}
      onClose={() => !busy && onClose()}
      title="Reshard collection"
      footer={
        <>
          <button type="button" className="btn-ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn-primary" onClick={submit} disabled={busy}>
            {busy ? 'Resharding…' : 'Reshard'}
          </button>
        </>
      }
    >
      <div className="space-y-3">
        <div className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
          This is an <strong>online</strong> reshard — the collection keeps serving reads
          and writes while partitions are rebuilt in the background.
        </div>
        <label className="block">
          <span className="mb-1 block text-xs font-medium text-muted">
            New partition count
          </span>
          <Badge tone="brand" mono>
            {name}
          </Badge>
          <input
            autoFocus
            className="input mt-2 font-mono"
            inputMode="numeric"
            placeholder="e.g. 4"
            value={parts}
            onChange={(e) => setParts(e.target.value.replace(/[^\d]/g, ''))}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
          />
        </label>
        {error && (
          <p className="rounded-lg border border-down/30 bg-down/10 px-3 py-2 font-mono text-xs text-down">
            {error}
          </p>
        )}
      </div>
    </Modal>
  );
}
