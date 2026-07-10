/** Extract human-readable detail from API error messages (API {status}: {body}). */
export function parseApiError(err: unknown, fallback = '操作失败'): string {
  if (!(err instanceof Error)) return fallback;
  const msg = err.message;
  const match = msg.match(/^API \d+: (.+)$/s);
  if (!match) return msg || fallback;
  const body = match[1].trim();
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed.detail === 'string') return parsed.detail;
    if (Array.isArray(parsed.detail)) {
      return parsed.detail
        .map((d: { msg?: string }) => d.msg)
        .filter(Boolean)
        .join('; ') || body;
    }
    if (parsed.detail && typeof parsed.detail === 'object') {
      return JSON.stringify(parsed.detail);
    }
  } catch {
    /* not json */
  }
  return body || fallback;
}
