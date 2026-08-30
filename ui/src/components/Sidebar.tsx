import { NavLink } from 'react-router-dom';
import {
  Boxes,
  Database,
  LayoutDashboard,
  Search,
  Server,
  ShieldCheck,
} from 'lucide-react';
import { classNames } from '../lib/format';

const nav = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/collections', label: 'Collections', icon: Boxes, end: false },
  { to: '/search', label: 'Search', icon: Search, end: false },
  { to: '/kv', label: 'KV', icon: Database, end: false },
  { to: '/admin', label: 'Admin', icon: ShieldCheck, end: false },
];

export function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-line bg-panel/60">
      <div className="flex h-14 items-center gap-2.5 border-b border-line px-4">
        <div className="grid h-7 w-7 place-items-center rounded-md bg-brand text-brand-ink shadow-sm">
          <Server size={16} strokeWidth={2.5} />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-tight text-ink">
            Rostam
          </div>
          <div className="text-2xs text-faint">vector + kv</div>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 p-2">
        {nav.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              classNames(
                'group relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-panel-2 text-ink'
                  : 'text-muted hover:bg-panel-2/60 hover:text-ink',
              )
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={classNames(
                    'absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-r-full bg-brand transition-opacity',
                    isActive ? 'opacity-100' : 'opacity-0',
                  )}
                />
                <Icon
                  size={16}
                  className={isActive ? 'text-brand' : 'text-faint group-hover:text-muted'}
                />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-line p-3 text-2xs text-faint">
        <p className="leading-relaxed">
          Self-hosted vector database &amp; sub-microsecond KV store.
        </p>
      </div>
    </aside>
  );
}
