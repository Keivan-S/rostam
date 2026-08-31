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
    // Hoisted so cleanup can abort an in-flight request, and reused as the
    // re-entrancy guard: tick() is self-scheduling (setTimeout, not
    // setInterval) so a slow/unreachable server can never let an older
    // response settle after a newer one — the next tick is scheduled only
    // once the current one finishes. Mirrors useMetrics.ts's polling loop.
    let ctrl: AbortController | undefined;

    const tick = async () => {
      ctrl = new AbortController();
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
      } finally {
        if (active) timer = window.setTimeout(tick, intervalMs);
      }
    };

    tick();
    return () => {
      active = false;
      ctrl?.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [intervalMs]);

  return { state, detail };
}
