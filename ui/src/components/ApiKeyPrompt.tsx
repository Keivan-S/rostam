import { useState } from 'react';
import { KeyRound, Lock } from 'lucide-react';
import { Modal } from './Modal';
import { useApiKey } from '../context/ApiKeyContext';

/**
 * Non-blocking prompt shown when the server answered 401. The user pastes an
 * API key; it is kept in sessionStorage (a secret) and used as a Bearer token.
 */
export function ApiKeyPrompt() {
  const { needsKey, setApiKey } = useApiKey();
  const [draft, setDraft] = useState('');

  const submit = () => {
    if (draft.trim()) setApiKey(draft.trim());
  };

  return (
    <Modal
      open={needsKey}
      onClose={() => {
        /* stays open until a key is provided or the user reloads */
      }}
      title="Authentication required"
      footer={
        <button
          type="button"
          className="btn-primary"
          onClick={submit}
          disabled={!draft.trim()}
        >
          Connect
        </button>
      }
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-lg bg-brand/10 p-2 text-brand">
          <Lock size={18} />
        </div>
        <div className="flex-1">
          <p className="text-sm text-muted">
            The server returned <span className="font-mono text-ink">401</span>.
            Enter an API key to continue.
          </p>
          <div className="relative mt-3">
            <KeyRound
              size={14}
              className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint"
            />
            <input
              autoFocus
              className="input pl-8 font-mono"
              type="password"
              placeholder="rostam_sk_…"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit()}
              autoComplete="off"
              spellCheck={false}
            />
          </div>
          <p className="mt-2 text-2xs leading-relaxed text-faint">
            Stored in sessionStorage — it clears when this tab closes. Sent as
            <span className="font-mono text-muted"> Authorization: Bearer …</span>.
          </p>
        </div>
      </div>
    </Modal>
  );
}
