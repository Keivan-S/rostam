import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { setApiKey as setClientKey } from '../api/client';

const STORAGE_KEY = 'rostam-api-key';

interface ApiKeyContextValue {
  apiKey: string | null;
  hasKey: boolean;
  /** True after a 401 was observed and no working key is set. */
  needsKey: boolean;
  setApiKey: (key: string | null) => void;
  clearApiKey: () => void;
  reportUnauthorized: () => void;
  /** Server target the dashboard is talking to (this origin). */
  serverTarget: string;
}

const ApiKeyContext = createContext<ApiKeyContextValue | null>(null);

// The key is a secret, so it lives in sessionStorage (clears when the tab
// closes) — deliberately NOT localStorage.
function readStored(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function ApiKeyProvider({ children }: { children: ReactNode }) {
  const [apiKey, setKeyState] = useState<string | null>(() => {
    const k = readStored();
    setClientKey(k);
    return k;
  });
  const [needsKey, setNeedsKey] = useState(false);

  const setApiKey = useCallback((key: string | null) => {
    const clean = key && key.trim() !== '' ? key.trim() : null;
    setClientKey(clean);
    setKeyState(clean);
    setNeedsKey(false);
    try {
      if (clean) sessionStorage.setItem(STORAGE_KEY, clean);
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // sessionStorage unavailable (private mode); the in-memory key still works.
    }
  }, []);

  const clearApiKey = useCallback(() => setApiKey(null), [setApiKey]);

  const reportUnauthorized = useCallback(() => setNeedsKey(true), []);

  // Keep the module-level client key in sync if the state was hydrated.
  useEffect(() => {
    setClientKey(apiKey);
  }, [apiKey]);

  const value = useMemo<ApiKeyContextValue>(
    () => ({
      apiKey,
      hasKey: apiKey !== null,
      needsKey,
      setApiKey,
      clearApiKey,
      reportUnauthorized,
      serverTarget:
        typeof window !== 'undefined' ? window.location.origin : '',
    }),
    [apiKey, needsKey, setApiKey, clearApiKey, reportUnauthorized],
  );

  return (
    <ApiKeyContext.Provider value={value}>{children}</ApiKeyContext.Provider>
  );
}

export function useApiKey(): ApiKeyContextValue {
  const ctx = useContext(ApiKeyContext);
  if (!ctx) throw new Error('useApiKey must be used within ApiKeyProvider');
  return ctx;
}
