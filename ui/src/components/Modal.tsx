import { useEffect, useRef, useState, type ReactNode } from 'react';
import { X } from 'lucide-react';

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  wide,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    // Focus the dialog for keyboard users.
    ref.current?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-4 pt-[10vh] backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`card w-full ${wide ? 'max-w-2xl' : 'max-w-md'} animate-fade-in shadow-pop outline-none`}
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close dialog"
            className="rounded-md p-1 text-faint transition-colors hover:bg-panel-2 hover:text-ink"
          >
            <X size={16} />
          </button>
        </div>
        <div className="px-5 py-4">{children}</div>
        {footer && (
          <div className="flex justify-end gap-2 border-t border-line px-5 py-3.5">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * A destructive-action modal that requires the user to type an exact
 * confirmation phrase (e.g. the collection name) before the action arms.
 */
export function TypedConfirmModal({
  open,
  onClose,
  onConfirm,
  title,
  phrase,
  confirmLabel = 'Delete',
  children,
  busy,
}: {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  phrase: string;
  confirmLabel?: string;
  children?: ReactNode;
  busy?: boolean;
}) {
  const [typed, setTyped] = useState('');
  const armed = typed === phrase;

  useEffect(() => {
    if (open) setTyped('');
  }, [open]);

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      footer={
        <>
          <button type="button" className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-danger"
            disabled={!armed || busy}
            onClick={onConfirm}
          >
            {busy ? 'Working…' : confirmLabel}
          </button>
        </>
      }
    >
      {children}
      <label className="mt-3 block text-xs text-muted">
        Type <span className="font-mono text-ink">{phrase}</span> to confirm
      </label>
      <input
        autoFocus
        className="input mt-1.5 font-mono"
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && armed && !busy) onConfirm();
        }}
        placeholder={phrase}
        spellCheck={false}
        autoComplete="off"
      />
    </Modal>
  );
}
