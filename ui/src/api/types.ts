// Response shapes for the Rostam HTTP API.
//
// NOTE: shapes are intentionally permissive. The live server reconciles some
// specifics at integration, and several endpoints (topology, collection list,
// KV) are being added by the backend concurrently. Fields that may be absent
// are optional, and generic config is kept as an index signature.

export interface ReadyResponse {
  status: string; // "ready" | "not ready"
  detail?: string;
}

export interface HealthResponse {
  status: string; // "ok"
}

export interface CollectionSummary {
  name: string;
  [k: string]: unknown;
}

export interface CollectionListResponse {
  collections: CollectionSummary[];
}

/** Generic collection config: known fields surfaced, everything else preserved. */
export interface CollectionConfig {
  dim?: number;
  metric?: string;
  m?: number;
  ef_construction?: number;
  ef_search?: number;
  quant?: string;
  persistent?: boolean;
  partitions?: number;
  index_type?: string;
  [k: string]: unknown;
}

export interface TopologyMember {
  node_id: string;
  server_addr: string;
}

export interface TopologyResponse {
  num_shards: number;
  members: TopologyMember[];
  leaders: string[]; // leader server_addr per shard index
  placement: string[][]; // per-shard list of node_ids
}

export interface BackupLag {
  node_id?: string;
  lag?: number;
  [k: string]: unknown;
}

export interface ReplicationShard {
  shard?: number;
  mode?: string;
  primary?: string;
  isr?: number;
  min_isr?: number;
  backups?: BackupLag[];
  [k: string]: unknown;
}

export interface ReplicationResponse {
  shards: ReplicationShard[];
}

/** One search hit. Plain KNN returns id+distance; docs/fusion add score+payload. */
export interface SearchHit {
  id: number;
  distance?: number;
  score?: number;
  payload?: Record<string, unknown> | null;
  content?: string;
  [k: string]: unknown;
}

export interface SearchResponse {
  results?: SearchHit[];
  documents?: SearchHit[];
  degraded?: boolean;
  missing?: unknown;
}

export interface CreateCollectionBody {
  name: string;
  config: CollectionConfig;
}

export interface KvGetResponse {
  found: boolean;
  value_b64?: string;
  value_utf8?: string;
  ttl_ms?: number;
}

export interface KvPutBody {
  value?: string;
  value_b64?: string;
  ttl_ms?: number;
}

export interface BackupsResponse {
  backups?: unknown[];
  error?: string;
}

export type HealthState = 'ready' | 'notready' | 'down' | 'unknown';
