import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

const STORAGE_KEY = 'rostam-poll-ms';
const DEFAULT_POLL_MS = 4000;
const MIN_POLL_MS = 1000;
const MAX_POLL_MS = 60000;

interface SettingsContextValue {
  pollMs: number;
  setPollMs: (ms: number) => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

function readStored(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const n = Number(raw);
      if (Number.isFinite(n)) return Math.min(MAX_POLL_MS, Math.max(MIN_POLL_MS, n));
    }
  } catch {
    /* ignore */
  }
  return DEFAULT_POLL_MS;
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [pollMs, setPollMsState] = useState<number>(readStored);

  const setPollMs = useCallback((ms: number) => {
    const clamped = Math.min(MAX_POLL_MS, Math.max(MIN_POLL_MS, Math.round(ms)));
    setPollMsState(clamped);
    try {
      localStorage.setItem(STORAGE_KEY, String(clamped));
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo(() => ({ pollMs, setPollMs }), [pollMs, setPollMs]);
  return (
    <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}
