// A tiny dependency-free SVG sparkline. Renders an area + line for a numeric
// series, scaling to its own min/max. Keeps the embedded bundle small.

import { useId } from 'react';

export function Sparkline({
  values,
  width = 120,
  height = 34,
  strokeWidth = 1.5,
  className,
  color = 'rgb(var(--c-brand))',
}: {
  values: number[];
  width?: number;
  height?: number;
  strokeWidth?: number;
  className?: string;
  color?: string;
}) {
  // useId() gives every Sparkline instance its own SVG id namespace, so two
  // charts with identical values (and thus identical hashSeries output) never
  // collide on the gradient id and steal each other's fill via url(#id).
  const reactId = useId();
  const pad = 2;
  const w = width - pad * 2;
  const h = height - pad * 2;

  if (values.length < 2) {
    return (
      <svg
        width={width}
        height={height}
        className={className}
        role="img"
        aria-label="sparkline (insufficient data)"
      >
        <line
          x1={pad}
          y1={height / 2}
          x2={width - pad}
          y2={height / 2}
          stroke="var(--c-line-strong)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
      </svg>
    );
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = w / (values.length - 1);

  const pts = values.map((v, i) => {
    const x = pad + i * step;
    const y = pad + h - ((v - min) / range) * h;
    return [x, y] as const;
  });

  const line = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const area =
    `${line} L${pts[pts.length - 1][0].toFixed(1)},${(height - pad).toFixed(1)}` +
    ` L${pts[0][0].toFixed(1)},${(height - pad).toFixed(1)} Z`;
  // useId()'s value contains colons, which are legal in an SVG id but awkward
  // in a url(#...) reference in some tooling, so they are stripped.
  const gid = `spark-${reactId.replace(/:/g, '')}-${Math.abs(hashSeries(values)).toString(36)}`;

  return (
    <svg
      width={width}
      height={height}
      className={className}
      role="img"
      aria-label="sparkline"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.22" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gid})`} />
      <path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle
        cx={pts[pts.length - 1][0]}
        cy={pts[pts.length - 1][1]}
        r={1.8}
        fill={color}
      />
    </svg>
  );
}

function hashSeries(values: number[]): number {
  let h = 0;
  for (let i = 0; i < values.length; i++) {
    h = (h * 31 + Math.round(values[i] * 1000)) | 0;
  }
  return h + values.length;
}
