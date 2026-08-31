# Rostam Dashboard

Embedded web UI for managing and monitoring a Rostam server (vector database +
sub-microsecond KV store). Built as a static SPA that the Go server serves under
the `/dashboard/` path prefix, talking to the same-origin HTTP API.

## Stack

- React 18 + TypeScript, Vite 5
- Tailwind CSS 3.4 (classic config)
- react-router-dom 6 (HashRouter — no server-side rewrite needed)
- lucide-react icons
- Hand-rolled SVG sparklines and a defensive Prometheus text parser (no chart
  or metrics dependencies) to keep the embedded bundle small

## Develop

```bash
npm install
npm run dev        # dev server on :5273, proxies /v1 and /metrics
```

Point the dev proxy at a running server with `ROSTAM_API`:

```bash
ROSTAM_API=http://127.0.0.1:8080 npm run dev
```

## Build

```bash
npm run build      # tsc type-check, then vite build → dist/
```

`vite.config.ts` sets `base: './'` so assets load with relative URLs under the
`/dashboard/` prefix. The output directory is `dist/`; the maintainer wires the
final embed path at integration.

## Layout

- `src/api/` — typed client (`client.ts`), endpoint wrappers (`endpoints.ts`),
  response types, and the Prometheus parser (`prom.ts`).
- `src/context/` — API-key (Bearer token, sessionStorage), theme, and settings.
- `src/hooks/` — `useMetrics` (polls `/metrics`, derives QPS/p99), `useHealth`
  (`/v1/ready` dot), `useAsync`.
- `src/components/` — layout chrome and reusable UI primitives.
- `src/pages/` — Overview, Collections, Search, KV, Admin.

## Auth

On load the app probes the auth-exempt `GET /v1/ready`. Data calls send
`Authorization: Bearer <key>` when a key is set. A `401` opens a non-blocking
prompt; the key is stored in **sessionStorage** (clears when the tab closes) —
never localStorage. Open (loopback, no-auth) servers work with no prompt.
