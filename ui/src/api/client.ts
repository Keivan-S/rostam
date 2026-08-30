// Typed same-origin API client for the Rostam server.
//
// Every data call goes through fetchJSON, which injects the Bearer header when
// an API key is present, parses the `{ "error": "..." }` JSON body the server
// returns on non-2xx, and throws an ApiError carrying the HTTP status so
// callers can branch on 401 (auth), 412/404 (feature unavailable), etc.

export class ApiError extends Error {
  status: number;
  /** Parsed server error string, when the body carried `{ "error": "..." }`. */
  serverMessage?: string;

  constructor(status: number, message: string, serverMessage?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.serverMessage = serverMessage;
  }

  get isAuth(): boolean {
    return this.status === 401 || this.status === 403;
  }

  /** The server has the route but the feature is not configured/enabled. */
  get isUnavailable(): boolean {
    return this.status === 412 || this.status === 404 || this.status === 501;
  }
}

/** Thrown when fetch itself fails (server down, CORS, network). */
export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NetworkError';
  }
}

// The current API key. Held in a module-level closure and mirrored to
// sessionStorage (a secret — deliberately NOT localStorage) by the key context.
let apiKey: string | null = null;

export function setApiKey(key: string | null): void {
  apiKey = key && key.trim() !== '' ? key.trim() : null;
}

export function getApiKey(): string | null {
  return apiKey;
}

function authHeaders(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  if (apiKey) h.set('Authorization', `Bearer ${apiKey}`);
  return h;
}

async function parseError(res: Response): Promise<ApiError> {
  let serverMessage: string | undefined;
  try {
    const body = await res.json();
    if (body && typeof body.error === 'string') serverMessage = body.error;
    else if (body && typeof body.detail === 'string') serverMessage = body.detail;
  } catch {
    // Non-JSON error body (e.g. a proxy 502). Fall through to the status text.
  }
  const message = serverMessage || res.statusText || `HTTP ${res.status}`;
  return new ApiError(res.status, message, serverMessage);
}

export interface RequestOptions {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  /** Skip auth header injection (auth-exempt probes like /v1/ready). */
  noAuth?: boolean;
}

/** Fetch + parse JSON, throwing ApiError on non-2xx and NetworkError on fetch failure. */
export async function fetchJSON<T>(
  path: string,
  opts: RequestOptions = {},
): Promise<T> {
  const headers = opts.noAuth ? new Headers() : authHeaders();
  let init: RequestInit = { method: opts.method || 'GET', headers, signal: opts.signal };
  if (opts.body !== undefined) {
    headers.set('Content-Type', 'application/json');
    init.body = JSON.stringify(opts.body);
  }

  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new NetworkError(
      e instanceof Error ? e.message : 'network request failed',
    );
  }

  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;

  const text = await res.text();
  if (text === '') return undefined as T;
  return JSON.parse(text) as T;
}

/** Fetch a text body (Prometheus exposition, etc.) with the same auth + error handling. */
export async function fetchText(
  path: string,
  opts: RequestOptions = {},
): Promise<string> {
  const headers = opts.noAuth ? new Headers() : authHeaders();
  let res: Response;
  try {
    res = await fetch(path, { method: opts.method || 'GET', headers, signal: opts.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new NetworkError(
      e instanceof Error ? e.message : 'network request failed',
    );
  }
  if (!res.ok) throw await parseError(res);
  return res.text();
}
