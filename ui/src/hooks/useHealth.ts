import { useEffect, useState } from 'react';
import { getReady } from '../api/endpoints';
import { NetworkError } from '../api/client';
import type { HealthState } from '../api/types';

/**
 * Poll the auth-exempt /v1/ready probe for the cluster health dot.
 * 200 -> ready, 503 -> notready, fetch failure -> down.
 */
export function useHealth(intervalMs = 5000): {
  state: HealthState;
  detail?: string;
} {
  const [state, setState] = useState<HealthState>('unknown');
  const [detail, setDetail] = useState<string | undefined>();

  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    const tick = async () => {
      const ctrl = new AbortController();
      try {
        const res = await getReady(ctrl.signal);
        if (!active) return;
        setState(res.status === 'ready' ? 'ready' : 'notready');
        setDetail(res.detail);
      } catch (e) {
        if (!active || (e instanceof DOMException && e.name === 'AbortError')) return;
        // 503 comes back as a parsed body via ApiError; a fetch failure is "down".
        if (e instanceof NetworkError) {
          setState('down');
          setDetail(e.message);
        } else {
          setState('notready');
          setDetail(e instanceof Error ? e.message : undefined);
        }
      }
    };

    tick();
    timer = window.setInterval(tick, intervalMs);
    return () => {
      active = false;
      if (timer) window.clearInterval(timer);
    };
  }, [intervalMs]);

  return { state, detail };
}
