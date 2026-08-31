import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { Modal } from './Modal';
import { createCollection } from '../api/endpoints';
import { ApiError } from '../api/client';
import type { CollectionConfig } from '../api/types';
import { classNames } from '../lib/format';

const METRICS = ['cosine', 'l2', 'dot'] as const;
const QUANTS = ['none', 'sq8', 'bq1', 'pq', 'sq', 'prq'] as const;

export function CreateCollectionForm({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (name: string) => void;
}) {
  const [name, setName] = useState('');
  const [dim, setDim] = useState('768');
  const [metric, setMetric] = useState<(typeof METRICS)[number]>('cosine');
  const [advanced, setAdvanced] = useState(false);
  const [m, setM] = useState('');
  const [efc, setEfc] = useState('');
  const [efs, setEfs] = useState('');
  const [quant, setQuant] = useState<(typeof QUANTS)[number]>('none');
  const [persistent, setPersistent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setName('');
    setDim('768');
    setMetric('cosine');
    setAdvanced(false);
    setM('');
    setEfc('');
    setEfs('');
    setQuant('none');
    setPersistent(false);
    setError(null);
  };

  const close = () => {
    if (!busy) {
      reset();
      onClose();
    }
  };

  const submit = async () => {
    setError(null);
    const dimNum = Number(dim);
    if (!name.trim()) return setError('Name is required.');
    if (!Number.isInteger(dimNum) || dimNum <= 0)
      return setError('Dimension must be a positive integer.');

    const config: CollectionConfig = { dim: dimNum, metric };
    if (quant !== 'none') config.quant = quant;
    if (persistent) config.persistent = true;
    if (advanced) {
      if (m) config.m = Number(m);
      if (efc) config.ef_construction = Number(efc);
      if (efs) config.ef_search = Number(efs);
    }

    setBusy(true);
    try {
      await createCollection({ name: name.trim(), config });
      setBusy(false);
      onCreated(name.trim());
      reset();
      onClose();
    } catch (e) {
      setBusy(false);
      setError(
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : 'Failed to create collection.',
      );
    }
  };

  return (
    <Modal
      open={open}
      onClose={close}
      title="Create collection"
      footer={
        <>
          <button type="button" className="btn-ghost" onClick={close} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn-primary" onClick={submit} disabled={busy}>
            {busy ? 'Creating…' : 'Create'}
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <Row label="Name">
          <input
            autoFocus
            className="input font-mono"
            placeholder="documents"
            value={name}
            onChange={(e) => setName(e.target.value)}
            spellCheck={false}
          />
        </Row>

        <div className="grid grid-cols-2 gap-3">
          <Row label="Dimension">
            <input
              className="input font-mono"
              inputMode="numeric"
              placeholder="768"
              value={dim}
              onChange={(e) => setDim(e.target.value.replace(/[^\d]/g, ''))}
            />
          </Row>
          <Row label="Metric">
            <Select value={metric} onChange={(v) => setMetric(v as (typeof METRICS)[number])} options={METRICS} />
          </Row>
        </div>

        <button
          type="button"
          onClick={() => setAdvanced((a) => !a)}
          className="flex items-center gap-1 text-xs font-medium text-muted hover:text-ink"
        >
          <ChevronDown
            size={13}
            className={classNames('transition-transform', advanced && 'rotate-180')}
          />
          Advanced index parameters
        </button>

        {advanced && (
          <div className="animate-fade-in space-y-4 rounded-lg border border-line bg-panel-2/50 p-3">
            <div className="grid grid-cols-3 gap-3">
              <Row label="M (degree)">
                <input className="input font-mono" placeholder="auto" value={m} onChange={(e) => setM(e.target.value.replace(/[^\d]/g, ''))} />
              </Row>
              <Row label="ef_construction">
                <input className="input font-mono" placeholder="auto" value={efc} onChange={(e) => setEfc(e.target.value.replace(/[^\d]/g, ''))} />
              </Row>
              <Row label="ef_search">
                <input className="input font-mono" placeholder="auto" value={efs} onChange={(e) => setEfs(e.target.value.replace(/[^\d]/g, ''))} />
              </Row>
            </div>
            <div className="grid grid-cols-2 items-end gap-3">
              <Row label="Quantization">
                <Select value={quant} onChange={(v) => setQuant(v as (typeof QUANTS)[number])} options={QUANTS} />
              </Row>
              <label className="flex cursor-pointer items-center gap-2 pb-2 text-xs text-muted">
                <input
                  type="checkbox"
                  checked={persistent}
                  onChange={(e) => setPersistent(e.target.checked)}
                  className="h-4 w-4 rounded border-line accent-brand"
                />
                Persistent (mmap-backed)
              </label>
            </div>
          </div>
        )}

        {error && (
          <p className="rounded-lg border border-down/30 bg-down/10 px-3 py-2 font-mono text-xs text-down">
            {error}
          </p>
        )}
      </div>
    </Modal>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-muted">{label}</span>
      {children}
    </label>
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: readonly string[];
}) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="input appearance-none pr-8 font-mono"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      <ChevronDown
        size={14}
        className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-faint"
      />
    </div>
  );
}
