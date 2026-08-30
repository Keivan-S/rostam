import { useRef, useState } from 'react';
import { Clock, Database, Info, Save, Search, Trash2 } from 'lucide-react';
import { PageHeader } from '../components/Layout';
import { Card, Badge, CopyButton } from '../components/primitives';
import { EmptyState } from '../components/StateView';
import { Modal } from '../components/Modal';
import { kvDelete, kvGet, kvPut } from '../api/endpoints';
import { ApiError } from '../api/client';
import { base64ToBytes, hexDump, utf8ToBytes } from '../lib/encoding';
import { classNames, formatTtl } from '../lib/format';
import type { KvGetResponse } from '../api/types';

type ValueView = 'utf8' | 'base64' | 'hex';

export function KVPage() {
  const [key, setKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<KvGetResponse | null>(null);
  const [notFoundKey, setNotFoundKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ValueView>('utf8');
  const [putOpen, setPutOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  // Tracks the most recently requested key, so a response from an older,
  // slower lookup (e.g. "a" resolving after a newer lookup for "b" already
  // started) can be told apart from the current one and discarded instead of
  // being rendered as — or acted on as — the wrong key's data.
  const latestKeyRef = useRef<string | null>(null);

  const doGet = async (k = key) => {
    const trimmed = k.trim();
    if (!trimmed) return;
    latestKeyRef.current = trimmed;
    setLoading(true);
    setError(null);
    setResult(null);
    setNotFoundKey(null);
    try {
      const res = await kvGet(trimmed);
      if (latestKeyRef.current !== trimmed) return;
      if (res.found) {
        setResult(res);
        setView(res.value_utf8 != null ? 'utf8' : 'base64');
      } else {
        setNotFoundKey(trimmed);
      }
    } catch (e) {
      if (latestKeyRef.current !== trimmed) return;
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : 'Lookup failed.');
    } finally {
      if (latestKeyRef.current === trimmed) setLoading(false);
    }
  };

  const bytes = result?.value_b64 ? base64ToBytes(result.value_b64) : null;

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Key-Value"
        description="Inspect and operate on individual keys in the KV store."
      />

      <div className="mb-4 flex items-start gap-2 rounded-lg border border-line bg-panel-2/60 px-3 py-2.5 text-xs text-muted">
        <Info size={14} className="mt-0.5 shrink-0 text-faint" />
        <span>
          The KV store has no key listing or scan — you can only operate on a key you
          name. Browsing all keys is not available yet.
        </span>
      </div>

      {/* Key bar */}
      <Card className="p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex-1">
            <span className="mb-1 block text-xs font-medium text-muted">Key</span>
            <div className="relative">
              <Database
                size={14}
                className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint"
              />
              <input
                className="input pl-8 font-mono"
                placeholder="session:abc123"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && doGet()}
                spellCheck={false}
              />
            </div>
          </div>
          <div className="flex gap-2">
            <button type="button" className="btn-primary" onClick={() => doGet()} disabled={loading || !key.trim()}>
              <Search size={14} /> {loading ? 'Loading…' : 'Get'}
            </button>
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setPutOpen(true)}
              disabled={!key.trim()}
            >
              <Save size={14} /> Set
            </button>
            <button
              type="button"
              className="btn-danger"
              onClick={() => setDeleteOpen(true)}
              disabled={!key.trim()}
              aria-label="Delete key"
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>

        {error && (
          <p className="mt-3 rounded-lg border border-down/30 bg-down/10 px-3 py-2 font-mono text-xs text-down">
            {error}
          </p>
        )}
      </Card>

      {/* Result */}
      <div className="mt-4">
        {notFoundKey ? (
          <EmptyState
            icon={<Database size={20} />}
            title="Key not found"
            hint={`No value is stored at "${notFoundKey}".`}
          />
        ) : result ? (
          <Card className="p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-ink">{key.trim()}</span>
                <Badge tone="ok">found</Badge>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted">
                <Clock size={13} className="text-faint" />
                {formatTtl(result.ttl_ms)}
              </div>
            </div>

            <div className="mb-2 flex items-center justify-between">
              <div className="inline-flex gap-1 rounded-lg border border-line bg-panel-2 p-0.5">
                {(['utf8', 'base64', 'hex'] as ValueView[]).map((v) => {
                  const disabled = v === 'utf8' && result.value_utf8 == null;
                  return (
                    <button
                      key={v}
                      type="button"
                      disabled={disabled}
                      onClick={() => setView(v)}
                      className={classNames(
                        'rounded-md px-2.5 py-1 text-2xs font-medium transition-colors',
                        view === v ? 'bg-panel text-ink shadow-sm' : 'text-muted hover:text-ink',
                        disabled && 'cursor-not-allowed opacity-40',
                      )}
                    >
                      {v === 'utf8' ? 'UTF-8' : v === 'base64' ? 'Base64' : 'Hex'}
                    </button>
                  );
                })}
              </div>
              <CopyButton
                value={currentText(view, result, bytes)}
                label="Copy"
              />
            </div>

            <pre className="max-h-96 overflow-auto rounded-lg border border-line bg-panel-2/60 p-3 font-mono text-xs text-ink">
              {currentText(view, result, bytes)}
            </pre>

            {bytes && (
              <p className="mt-2 text-2xs text-faint">{bytes.length} bytes</p>
            )}
          </Card>
        ) : (
          <EmptyState
            icon={<Search size={20} />}
            title="Look up a key"
            hint="Enter a key and hit Get to inspect its value, encoding, and TTL."
          />
        )}
      </div>

      <PutModal
        open={putOpen}
        keyName={key.trim()}
        onClose={() => setPutOpen(false)}
        onDone={() => {
          setPutOpen(false);
          doGet();
        }}
      />

      <DeleteModal
        open={deleteOpen}
        keyName={key.trim()}
        onClose={() => setDeleteOpen(false)}
        onDone={() => {
          setDeleteOpen(false);
          setResult(null);
          setNotFoundKey(key.trim());
        }}
      />
    </div>
  );
}

function currentText(view: ValueView, res: KvGetResponse, bytes: Uint8Array | null): string {
  if (view === 'utf8') return res.value_utf8 ?? '(not valid UTF-8)';
  if (view === 'base64') return res.value_b64 ?? '';
  if (!bytes) return '';
  return hexDump(bytes);
}

// --- Set modal ---------------------------------------------------------------

function PutModal({
  open,
  keyName,
  onClose,
  onDone,
}: {
  open: boolean;
  keyName: string;
  onClose: () => void;
  onDone: () => void;
}) {
  const [encoding, setEncoding] = useState<'utf8' | 'base64'>('utf8');
  const [value, setValue] = useState('');
  const [ttl, setTtl] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setError(null);
    if (encoding === 'base64' && base64ToBytes(value) === null)
      return setError('Value is not valid base64.');

    setBusy(true);
    try {
      const body: { value?: string; value_b64?: string; ttl_ms?: number } =
        encoding === 'utf8' ? { value } : { value_b64: value };
      if (ttl.trim()) body.ttl_ms = Number(ttl);
      await kvPut(keyName, body);
      setBusy(false);
      setValue('');
      setTtl('');
      onDone();
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : 'Set failed.');
    }
  };

  const byteLen =
    encoding === 'utf8'
      ? utf8ToBytes(value).length
      : base64ToBytes(value)?.length ?? 0;

  return (
    <Modal
      open={open}
      onClose={() => !busy && onClose()}
      title="Set value"
      footer={
        <>
          <button type="button" className="btn-ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn-primary" onClick={submit} disabled={busy || !keyName}>
            {busy ? 'Saving…' : 'Save'}
          </button>
        </>
      }
    >
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">key</span>
          <Badge tone="brand" mono>
            {keyName || '(none)'}
          </Badge>
        </div>

        <div className="inline-flex gap-1 rounded-lg border border-line bg-panel-2 p-0.5">
          {(['utf8', 'base64'] as const).map((e) => (
            <button
              key={e}
              type="button"
              onClick={() => setEncoding(e)}
              className={classNames(
                'rounded-md px-2.5 py-1 text-2xs font-medium transition-colors',
                encoding === e ? 'bg-panel text-ink shadow-sm' : 'text-muted hover:text-ink',
              )}
            >
              {e === 'utf8' ? 'UTF-8' : 'Base64'}
            </button>
          ))}
        </div>

        <textarea
          className="input h-28 resize-y font-mono text-xs"
          placeholder={encoding === 'utf8' ? 'value text…' : 'base64…'}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          spellCheck={false}
        />
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs text-muted">
            TTL (ms)
            <input
              className="input w-32 font-mono"
              inputMode="numeric"
              placeholder="none"
              value={ttl}
              onChange={(e) => setTtl(e.target.value.replace(/[^\d]/g, ''))}
            />
          </label>
          <span className="font-mono text-2xs text-faint">{byteLen} bytes</span>
        </div>

        {error && (
          <p className="rounded-lg border border-down/30 bg-down/10 px-3 py-2 font-mono text-xs text-down">
            {error}
          </p>
        )}
      </div>
    </Modal>
  );
}

// --- Delete modal ------------------------------------------------------------

function DeleteModal({
  open,
  keyName,
  onClose,
  onDone,
}: {
  open: boolean;
  keyName: string;
  onClose: () => void;
  onDone: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await kvDelete(keyName);
      setBusy(false);
      onDone();
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : 'Delete failed.');
    }
  };

  return (
    <Modal
      open={open}
      onClose={() => !busy && onClose()}
      title="Delete key"
      footer={
        <>
          <button type="button" className="btn-ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn-danger" onClick={submit} disabled={busy}>
            {busy ? 'Deleting…' : 'Delete'}
          </button>
        </>
      }
    >
      <p className="text-sm text-muted">
        Delete the key <span className="font-mono text-ink">{keyName}</span>? This
        cannot be undone.
      </p>
      {error && (
        <p className="mt-3 rounded-lg border border-down/30 bg-down/10 px-3 py-2 font-mono text-xs text-down">
          {error}
        </p>
      )}
    </Modal>
  );
}
