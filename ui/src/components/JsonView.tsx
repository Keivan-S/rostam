import { useState } from 'react';
import { ChevronRight } from 'lucide-react';
import { classNames } from '../lib/format';

// A compact, syntax-tinted JSON viewer with collapsible objects/arrays.
export function JsonView({
  data,
  initialCollapsed = false,
}: {
  data: unknown;
  initialCollapsed?: boolean;
}) {
  return (
    <div className="font-mono text-xs leading-relaxed">
      <Node value={data} depth={0} collapsed={initialCollapsed} />
    </div>
  );
}

function Node({
  value,
  depth,
  keyName,
  collapsed,
}: {
  value: unknown;
  depth: number;
  keyName?: string;
  collapsed?: boolean;
}) {
  const [open, setOpen] = useState(!collapsed || depth < 1);
  const prefix = keyName !== undefined && (
    <span className="text-info">"{keyName}"</span>
  );

  if (value === null) return <Leaf prefix={prefix} className="text-faint">null</Leaf>;
  if (typeof value === 'string')
    return (
      <Leaf prefix={prefix} className="text-ok break-all">
        "{value}"
      </Leaf>
    );
  if (typeof value === 'number')
    return <Leaf prefix={prefix} className="text-brand">{String(value)}</Leaf>;
  if (typeof value === 'boolean')
    return <Leaf prefix={prefix} className="text-warn">{String(value)}</Leaf>;

  const isArray = Array.isArray(value);
  const entries = isArray
    ? (value as unknown[]).map((v, i) => [String(i), v] as const)
    : Object.entries(value as Record<string, unknown>);
  const open1 = isArray ? '[' : '{';
  const close1 = isArray ? ']' : '}';

  if (entries.length === 0)
    return (
      <Leaf prefix={prefix} className="text-faint">
        {open1}
        {close1}
      </Leaf>
    );

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-0.5 hover:text-ink"
      >
        <ChevronRight
          size={11}
          className={classNames('text-faint transition-transform', open && 'rotate-90')}
        />
        {prefix}
        {keyName !== undefined && <span className="text-faint">: </span>}
        <span className="text-faint">
          {open1}
          {!open && (
            <span className="text-faint">
              {' '}
              {entries.length} {isArray ? 'items' : 'keys'}{' '}
              {close1}
            </span>
          )}
        </span>
      </button>
      {open && (
        <div className="border-l border-line pl-3" style={{ marginLeft: 5 }}>
          {entries.map(([k, v]) => (
            <div key={k}>
              <Node value={v} depth={depth + 1} keyName={isArray ? undefined : k} />
            </div>
          ))}
        </div>
      )}
      {open && <div className="text-faint">{close1}</div>}
    </div>
  );
}

function Leaf({
  prefix,
  className,
  children,
}: {
  prefix?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span>
      {prefix}
      {prefix && <span className="text-faint">: </span>}
      <span className={className}>{children}</span>
    </span>
  );
}
