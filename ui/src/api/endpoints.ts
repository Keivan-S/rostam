// Typed endpoint wrappers over fetchJSON. Paths are same-origin relative so the
// dashboard talks to whichever server serves it.

import { fetchJSON, fetchText } from './client';
import type {
  BackupsResponse,
  CollectionConfig,
  CollectionListResponse,
  CreateCollectionBody,
  KvGetResponse,
  KvPutBody,
  ReadyResponse,
  ReplicationResponse,
  SearchResponse,
  TopologyResponse,
} from './types';

// --- Observability (auth-exempt probes) --------------------------------------

export const getReady = (signal?: AbortSignal) =>
  fetchJSON<ReadyResponse>('/v1/ready', { noAuth: true, signal });

export const getMetricsText = (signal?: AbortSignal) =>
  fetchText('/metrics', { signal });

export const getReplication = (signal?: AbortSignal) =>
  fetchJSON<ReplicationResponse>('/v1/replication', { noAuth: true, signal });

// /v1/topology is scope-gated (served through the authed op path, unlike the
// exempt /v1/ready and /v1/replication probes), so it must carry the API key
// when one is configured — otherwise the cluster view 401s under auth.
export const getTopology = (signal?: AbortSignal) =>
  fetchJSON<TopologyResponse>('/v1/topology', { signal });

// --- Collections -------------------------------------------------------------

export const listCollections = (signal?: AbortSignal) =>
  fetchJSON<CollectionListResponse>('/v1/collections', { signal });

export const getCollection = (name: string, signal?: AbortSignal) =>
  fetchJSON<CollectionConfig>(`/v1/collections/${encodeURIComponent(name)}`, { signal });

export const createCollection = (body: CreateCollectionBody) =>
  fetchJSON<{ name: string }>('/v1/collections', { method: 'POST', body });

export const dropCollection = (name: string) =>
  fetchJSON<{ dropped: string }>(`/v1/collections/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });

export const reshardCollection = (name: string, newPartitions: number) =>
  fetchJSON<{ name: string; new_partitions: number }>(
    `/v1/collections/${encodeURIComponent(name)}/reshard`,
    { method: 'POST', body: { new_partitions: newPartitions } },
  );

// --- Search ------------------------------------------------------------------

export const vectorSearch = (
  name: string,
  query: number[],
  k: number,
  filter: unknown | null,
) =>
  fetchJSON<SearchResponse>(
    `/v1/collections/${encodeURIComponent(name)}/points/search`,
    { method: 'POST', body: { query, k, filter } },
  );

// The server's text-search body uses `text` (not `query`) and returns
// `documents`. A 400 with "full text disabled" means the collection has no BM25.
export const textSearch = (
  name: string,
  text: string,
  k: number,
  filter: unknown | null,
) =>
  fetchJSON<SearchResponse>(
    `/v1/collections/${encodeURIComponent(name)}/points/search/text`,
    { method: 'POST', body: { text, k, filter } },
  );

// --- KV ----------------------------------------------------------------------

export const kvGet = (key: string, signal?: AbortSignal) =>
  fetchJSON<KvGetResponse>(
    `/v1/kv/${encodeURIComponent(key)}?with_ttl=1`,
    { signal },
  );

export const kvPut = (key: string, body: KvPutBody) =>
  fetchJSON<unknown>(`/v1/kv/${encodeURIComponent(key)}`, {
    method: 'PUT',
    body,
  });

export const kvDelete = (key: string) =>
  fetchJSON<unknown>(`/v1/kv/${encodeURIComponent(key)}`, { method: 'DELETE' });

// --- Admin -------------------------------------------------------------------

export const triggerBackup = () =>
  fetchJSON<BackupsResponse>('/v1/admin/backup', { method: 'POST' });

export const listBackups = (signal?: AbortSignal) =>
  fetchJSON<BackupsResponse>('/v1/admin/backups', { signal });
