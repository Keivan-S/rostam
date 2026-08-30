import { useState } from 'react';
import { Archive, DatabaseBackup, HardDriveDownload, RefreshCw } from 'lucide-react';
import { PageHeader } from '../components/Layout';
import { Card, SectionTitle } from '../components/primitives';
import { EmptyState, ErrorCard } from '../components/StateView';
import { JsonView } from '../components/JsonView';
import { useAsync } from '../hooks/useAsync';
import { listBackups, triggerBackup } from '../api/endpoints';
import { ApiError } from '../api/client';
import type { BackupsResponse } from '../api/types';

export function Admin() {
  const backups = useAsync<BackupsResponse>((s) => listBackups(s), []);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<{ tone: 'ok' | 'down'; text: string } | null>(null);

  const unavailable = backups.error instanceof ApiError && backups.error.isUnavailable;

  const runBackup = async () => {
    setRunning(true);
    setMessage(null);
    try {
      const res = await triggerBackup();
      setMessage(
        res.error
          ? { tone: 'down', text: `Backup completed with errors: ${res.error}` }
          : { tone: 'ok', text: 'Backup triggered successfully.' },
      );
      backups.reload();
    } catch (e) {
      const text =
        e instanceof ApiError && e.isUnavailable
          ? 'Object storage is not configured on this server.'
          : e instanceof Error
            ? e.message
            : 'Backup failed.';
      setMessage({ tone: 'down', text });
    } finally {
      setRunning(false);
    }
  };

  const items = backups.data?.backups ?? [];

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Admin"
        description="Object-storage backups and disaster-recovery snapshots."
        actions={
          <button type="button" className="btn-ghost" onClick={() => backups.reload()} aria-label="Refresh">
            <RefreshCw size={14} />
          </button>
        }
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <Card className="p-4 lg:col-span-1">
          <SectionTitle title="Trigger backup" subtitle="Snapshot every collection to object storage." />
          <div className="mt-4 flex flex-col items-center gap-3 rounded-lg border border-dashed border-line bg-panel-2/40 px-4 py-6 text-center">
            <div className="rounded-xl bg-brand/10 p-3 text-brand">
              <DatabaseBackup size={22} />
            </div>
            <button type="button" className="btn-primary" onClick={runBackup} disabled={running}>
              <HardDriveDownload size={15} />
              {running ? 'Backing up…' : 'Back up now'}
            </button>
          </div>
          {message && (
            <p
              className={
                'mt-3 rounded-lg border px-3 py-2 text-xs ' +
                (message.tone === 'ok'
                  ? 'border-ok/30 bg-ok/10 text-ok'
                  : 'border-down/30 bg-down/10 text-down')
              }
            >
              {message.text}
            </p>
          )}
        </Card>

        <Card className="p-4 lg:col-span-2">
          <SectionTitle
            title="Backups"
            subtitle="Snapshot objects under the configured tenant prefix."
          />
          <div className="mt-4">
            {backups.error ? (
              unavailable ? (
                <EmptyState
                  icon={<Archive size={20} />}
                  title="Backups not configured"
                  hint="This server has no object-storage backend configured, so backup and restore are unavailable."
                />
              ) : backups.loading ? null : (
                <ErrorCard error={backups.error} onRetry={backups.reload} />
              )
            ) : !backups.data ? (
              <div className="h-24 animate-pulse rounded-lg bg-panel-2" />
            ) : items.length === 0 ? (
              <EmptyState
                icon={<Archive size={20} />}
                title="No backups yet"
                hint="Trigger a backup to create the first snapshot."
              />
            ) : (
              <div className="overflow-x-auto rounded-lg border border-line bg-panel-2/40 p-3">
                <JsonView data={items} initialCollapsed />
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
