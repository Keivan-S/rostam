// Base64 / UTF-8 / hex conversions for the KV value viewer. All are tolerant:
// they return null rather than throwing on invalid input.

export function base64ToBytes(b64: string): Uint8Array | null {
  try {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  } catch {
    return null;
  }
}

export function bytesToBase64(bytes: Uint8Array): string {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

export function utf8ToBytes(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

/** Decode bytes as UTF-8, returning null if the bytes are not valid UTF-8. */
export function bytesToUtf8(bytes: Uint8Array): string | null {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return null;
  }
}

export function bytesToHex(bytes: Uint8Array): string {
  let out = '';
  for (let i = 0; i < bytes.length; i++) {
    out += bytes[i].toString(16).padStart(2, '0');
    if (i < bytes.length - 1) out += i % 2 === 1 ? ' ' : '';
  }
  return out;
}

/** A canonical hex dump (offset | hex | ascii), 16 bytes per row. */
export function hexDump(bytes: Uint8Array): string {
  const rows: string[] = [];
  for (let off = 0; off < bytes.length; off += 16) {
    const slice = bytes.subarray(off, off + 16);
    const hex = Array.from(slice)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join(' ')
      .padEnd(16 * 3 - 1, ' ');
    const ascii = Array.from(slice)
      .map((b) => (b >= 0x20 && b < 0x7f ? String.fromCharCode(b) : '.'))
      .join('');
    rows.push(`${off.toString(16).padStart(8, '0')}  ${hex}  ${ascii}`);
  }
  return rows.join('\n');
}
