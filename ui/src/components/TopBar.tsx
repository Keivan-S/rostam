import { Moon, Settings, Sun } from 'lucide-react';
import { HealthDot } from './HealthDot';
import { useHealth } from '../hooks/useHealth';
import { useTheme } from '../context/ThemeContext';
import { useApiKey } from '../context/ApiKeyContext';

export function TopBar({ onOpenSettings }: { onOpenSettings: () => void }) {
  const { state, detail } = useHealth();
  const { theme, toggle } = useTheme();
  const { serverTarget, hasKey } = useApiKey();
  const host = serverTarget.replace(/^https?:\/\//, '') || 'localhost';

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-line bg-panel/60 px-4 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 rounded-lg border border-line bg-panel px-2.5 py-1.5">
          <HealthDot state={state} detail={detail} />
        </div>
        <div className="hidden items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 sm:flex">
          <span className="text-2xs text-faint">target</span>
          <span className="font-mono text-xs text-ink">{host}</span>
          {hasKey && (
            <span className="ml-1 rounded bg-brand/10 px-1 py-0.5 text-2xs font-medium text-brand">
              authed
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={toggle}
          className="grid h-9 w-9 place-items-center rounded-lg border border-line bg-panel text-muted transition-colors hover:text-ink"
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        <button
          type="button"
          onClick={onOpenSettings}
          className="grid h-9 w-9 place-items-center rounded-lg border border-line bg-panel text-muted transition-colors hover:text-ink"
          aria-label="Settings"
        >
          <Settings size={16} />
        </button>
      </div>
    </header>
  );
}
