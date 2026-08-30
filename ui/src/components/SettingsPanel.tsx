import { useState } from 'react';
import { Eye, EyeOff, KeyRound, Trash2 } from 'lucide-react';
import { Modal } from './Modal';
import { Badge } from './primitives';
import { useApiKey } from '../context/ApiKeyContext';
import { useSettings } from '../context/SettingsContext';

export function SettingsPanel({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { apiKey, hasKey, setApiKey, clearApiKey, serverTarget } = useApiKey();
  const { pollMs, setPollMs } = useSettings();
  const [draft, setDraft] = useState('');
  const [reveal, setReveal] = useState(false);

  const save = () => {
    if (draft.trim()) {
      setApiKey(draft.trim());
      setDraft('');
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Settings">
      <div className="space-y-5">
        <Field label="Server target">
          <div className="flex items-center gap-2">
            <code className="flex-1 truncate rounded-lg border border-line bg-panel-2 px-3 py-2 font-mono text-xs text-ink">
              {serverTarget || '—'}
            </code>
            <Badge tone="neutral">same origin</Badge>
          </div>
        </Field>

        <Field
          label="API key"
          hint="Sent as a Bearer token. Stored in sessionStorage — it clears when this tab closes."
        >
          {hasKey ? (
            <div className="flex items-center gap-2">
              <code className="flex-1 truncate rounded-lg border border-line bg-panel-2 px-3 py-2 font-mono text-xs text-ink">
                {reveal ? apiKey : '•'.repeat(Math.min(24, (apiKey || '').length))}
              </code>
              <button
                type="button"
                className="btn-ghost !px-2"
                onClick={() => setReveal((r) => !r)}
                aria-label={reveal ? 'Hide key' : 'Reveal key'}
              >
                {reveal ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
              <button
                type="button"
                className="btn-danger !px-2"
                onClick={clearApiKey}
                aria-label="Clear key"
              >
                <Trash2 size={15} />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <KeyRound
                  size={14}
                  className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint"
                />
                <input
                  className="input pl-8 font-mono"
                  type="password"
                  placeholder="rostam_sk_…"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && save()}
                  autoComplete="off"
                />
              </div>
              <button type="button" className="btn-primary" onClick={save} disabled={!draft.trim()}>
                Save
              </button>
            </div>
          )}
        </Field>

        <Field label="Refresh interval" hint="How often live metrics are polled.">
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={1}
              max={30}
              step={1}
              value={Math.round(pollMs / 1000)}
              onChange={(e) => setPollMs(Number(e.target.value) * 1000)}
              className="flex-1 accent-brand"
            />
            <span className="w-12 text-right font-mono text-sm text-ink tabular-nums">
              {(pollMs / 1000).toFixed(0)}s
            </span>
          </div>
        </Field>
      </div>
    </Modal>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1.5 text-xs font-medium text-ink">{label}</div>
      {children}
      {hint && <p className="mt-1.5 text-2xs leading-relaxed text-faint">{hint}</p>}
    </div>
  );
}
