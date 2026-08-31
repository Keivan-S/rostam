import { useMemo, useState } from 'react';
import { Play, Search as SearchIcon, Type } from 'lucide-react';
import { PageHeader } from '../components/Layout';
import { Card } from '../components/primitives';
import { EmptyState } from '../components/StateView';
import { JsonView } from '../components/JsonView';
import { useMetrics } from '../hooks/useMetrics';
import { useAsync } from '../hooks/useAsync';
import { listCollections, textSearch, vectorSearch } from '../api/endpoints';
import { ApiError } from '../api/client';
import { classNames } from '../lib/format';
import type { CollectionListResponse, SearchHit, SearchResponse } from '../api/types';

type Mode = 'vector' | 'text';

export function SearchPage() {
  const list = useAsync<CollectionListResponse>((s) => listCollections(s), []);
  const { snapshot } = useMetrics();

  const collectionNames = useMemo(() => {
    const names = new Set<string>();
    list.data?.collections?.forEach((c) => names.add(c.name));
    snapshot?.collections.forEach((c) => names.add(c.name));
    return Array.from(names).sort((a, b) => a.localeCompare(b));
  }, [list.data, snapshot]);

  const [mode, setMode] = useState<Mode>('vector');
  const [collection, setCollection] = useState('');
  const [vectorText, setVectorText] = useState('[0.1, 0.2, 0.3]');
  const [queryText, setQueryText] = useState('');
  const [k, setK] = useState('10');
  const [filterText, setFilterText] = useState('');

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tookMs, setTookMs] = useState<number | null>(null);

  const activeCollection = collection || collectionNames[0] || '';

  const run = async () => {
    setError(null);
    setResult(null);
    setTookMs(null);
    if (!activeCollection) return setError('Select a collection.');

    const kNum = Number(k) || 10;

    let filter: unknown | null = null;
    if (filterText.trim()) {
      try {
        filter = JSON.parse(filterText);
      } catch {
        return setError('Filter is not valid JSON.');
      }
    }

    let body: number[] | undefined;
    if (mode === 'vector') {
      try {
        const parsed = JSON.parse(vectorText);
        if (!Array.isArray(parsed) || !parsed.every((n) => typeof n === 'number'))
          throw new Error();
        body = parsed as number[];
      } catch {
        return setError('Query vector must be a JSON array of numbers, e.g. [0.1, 0.2].');
      }
    } else if (!queryText.trim()) {
      return setError('Enter query text.');
    }

    setRunning(true);
    const started = performance.now();
    try {
      const res =
        mode === 'vector'
          ? await vectorSearch(activeCollection, body!, kNum, filter)
          : await textSearch(activeCollection, queryText, kNum, filter);
      setResult(res);
      setTookMs(performance.now() - started);
    } catch (e) {
      if (
        mode === 'text' &&
        e instanceof ApiError &&
        /full[\s-]?text|text.*disabl/i.test(e.message)
      ) {
        setError(
          'Full-text search is disabled for this collection. It must be created with a BM25 text index.',
        );
      } else {
        setError(e instanceof Error ? e.message : 'Search failed.');
      }
    } finally {
      setRunning(false);
    }
  };

  const hits: SearchHit[] = result?.results || result?.documents || [];

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Search"
        description="Run vector or full-text queries against a collection."
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[380px_1fr]">
        {/* Query panel */}
        <Card className="h-fit p-4">
          <div className="mb-4 grid grid-cols-2 gap-1 rounded-lg border border-line bg-panel-2 p-1">
            <TabButton active={mode === 'vector'} onClick={() => setMode('vector')} icon={<SearchIcon size={13} />}>
              Vector
            </TabButton>
            <TabButton active={mode === 'text'} onClick={() => setMode('text')} icon={<Type size={13} />}>
              Text
            </TabButton>
          </div>

          <Label>Collection</Label>
          {collectionNames.length === 0 ? (
            <input
              className="input font-mono"
              placeholder="collection name"
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
            />
          ) : (
            <select
              className="input font-mono"
              value={activeCollection}
              onChange={(e) => setCollection(e.target.value)}
            >
              {collectionNames.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          )}

          {mode === 'vector' ? (
            <>
              <Label className="mt-3">Query vector (JSON array)</Label>
              <textarea
                className="input h-24 resize-y font-mono text-xs"
                value={vectorText}
                onChange={(e) => setVectorText(e.target.value)}
                spellCheck={false}
                placeholder="[0.12, -0.03, 0.98, …]"
              />
            </>
          ) : (
            <>
              <Label className="mt-3">Query text</Label>
              <textarea
                className="input h-24 resize-y text-sm"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="search phrase…"
              />
            </>
          )}

          <div className="mt-3 grid grid-cols-3 gap-3">
            <div className="col-span-1">
              <Label>k</Label>
              <input
                className="input font-mono"
                inputMode="numeric"
                value={k}
                onChange={(e) => setK(e.target.value.replace(/[^\d]/g, ''))}
              />
            </div>
          </div>

          <Label className="mt-3">Filter (optional JSON)</Label>
          <textarea
            className="input h-20 resize-y font-mono text-xs"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            spellCheck={false}
            placeholder='{"must": [{"key": "lang", "match": {"value": "en"}}]}'
          />

          <button
            type="button"
            className="btn-primary mt-4 w-full"
            onClick={run}
            disabled={running}
          >
            <Play size={14} /> {running ? 'Searching…' : 'Run search'}
          </button>

          {error && (
            <p className="mt-3 rounded-lg border border-down/30 bg-down/10 px-3 py-2 font-mono text-xs text-down">
              {error}
            </p>
          )}
        </Card>

        {/* Results panel */}
        <div>
          <div className="mb-2 flex items-center justify-between px-1">
            <span className="text-xs font-medium text-muted">
              {hits.length > 0 ? `${hits.length} result${hits.length === 1 ? '' : 's'}` : 'Results'}
            </span>
            <div className="flex items-center gap-2">
              {result?.degraded && (
                <span className="rounded-md border border-warn/30 bg-warn/10 px-1.5 py-0.5 text-2xs text-warn">
                  degraded
                </span>
              )}
              {tookMs != null && (
                <span className="font-mono text-2xs text-faint">{tookMs.toFixed(1)} ms</span>
              )}
            </div>
          </div>

          {hits.length === 0 ? (
            <EmptyState
              icon={<SearchIcon size={20} />}
              title={result ? 'No matches' : 'Run a query to see results'}
              hint={
                result
                  ? 'The query returned no hits for this collection.'
                  : 'Results appear here as a ranked table with expandable payloads.'
              }
            />
          ) : (
            <Card className="overflow-hidden">
              <div className="divide-y divide-line/60">
                {hits.map((hit, i) => (
                  <ResultRow key={`${hit.id}-${i}`} rank={i + 1} hit={hit} />
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultRow({ rank, hit }: { rank: number; hit: SearchHit }) {
  const [open, setOpen] = useState(false);
  const score = hit.score ?? hit.distance;
  const scoreLabel = hit.score != null ? 'score' : hit.distance != null ? 'distance' : '';
  const payload = hit.payload ?? (hit.content ? { content: hit.content } : null);

  return (
    <div className="px-4 py-3">
      <div
        className="flex cursor-pointer items-center gap-3"
        onClick={() => payload && setOpen((o) => !o)}
      >
        <span className="w-6 shrink-0 text-right font-mono text-2xs text-faint">{rank}</span>
        <span className="font-mono text-sm text-ink">#{hit.id}</span>
        <div className="flex-1" />
        {score != null && (
          <span className="font-mono text-xs text-muted">
            <span className="text-faint">{scoreLabel} </span>
            {typeof score === 'number' ? score.toFixed(4) : String(score)}
          </span>
        )}
      </div>
      {payload && (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="mt-1 pl-9 text-2xs text-faint hover:text-muted"
        >
          {open ? 'hide payload' : 'show payload'}
        </button>
      )}
      {open && payload && (
        <div className="mt-2 overflow-x-auto rounded-lg border border-line bg-panel-2/60 p-3 pl-9">
          <JsonView data={payload} />
        </div>
      )}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={classNames(
        'flex items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-colors',
        active ? 'bg-panel text-ink shadow-sm' : 'text-muted hover:text-ink',
      )}
    >
      {icon}
      {children}
    </button>
  );
}

function Label({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={classNames('mb-1 block text-xs font-medium text-muted', className)}>
      {children}
    </span>
  );
}
