// Date formatting that reads the calendar date straight from the ISO string.
//
// Clinical dates land in the DB at UTC midnight (e.g. "2024-08-12T00:00:00+00:00").
// `new Date(iso)` then renders the *previous* day in negative-offset zones, so a
// reading recorded on Aug 12 prints "Aug 11". We slice the YYYY-MM-DD portion and
// format that directly — faithful to the date the source recorded, no TZ drift.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function parts(iso: string | null | undefined): [number, number, number] | null {
  if (!iso) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!m) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

/** "Aug 12 '24" */
export function fmtShort(iso: string | null | undefined): string {
  const p = parts(iso);
  if (!p) return "";
  const [y, mo, d] = p;
  return `${MONTHS[mo - 1]} ${d} '${String(y).slice(2)}`;
}

/** "Aug 12, 2024" */
export function fmtDay(iso: string | null | undefined): string {
  const p = parts(iso);
  if (!p) return "";
  const [y, mo, d] = p;
  return `${MONTHS[mo - 1]} ${d}, ${y}`;
}

/** Four-digit year, or null. */
export function yearOf(iso: string | null | undefined): number | null {
  const p = parts(iso);
  return p ? p[0] : null;
}
