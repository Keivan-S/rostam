import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from '../api/client';
import { useApiKey } from '../context/ApiKeyContext';

export interface AsyncState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  /** Re-run the loader. */
  reload: () => void;
}

/**
 * Run an async loader on mount and whenever a dep changes. Aborts the in-flight
 * request on unmount/reload, and routes a 401 to the API-key prompt.
 */
export function useAsync<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const { apiKey, reportUnauthorized } = useApiKey();
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    const ctrl = new AbortController();
    let active = true;
    setLoading(true);
    setError(null);
    loaderRef
      .current(ctrl.signal)
      .then((res) => {
        if (!active) return;
        setData(res);
        setError(null);
      })
      .catch((e) => {
        if (!active || ctrl.signal.aborted) return;
        if (e instanceof ApiError && e.status === 401) reportUnauthorized();
        setError(e instanceof Error ? e : new Error(String(e)));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce, apiKey]);

  return { data, error, loading, reload };
}
