import { AppError, getErrorMessage } from '@flow/shared';

/**
 * Web search as a first-class pipeline input.
 *
 * The OpenCode dev engine's one-shot endpoint cannot run provider tools, and
 * paid engines only search natively when they do. So the executor treats
 * search as data: it searches for the target here, then hands the results to
 * the model inside the prompt — which keeps research grounded no matter which
 * engine answers, and costs a single search call per step.
 *
 * Tavily is the provider: a free-tier search API authenticated with a Bearer
 * key from TAVILY_API_KEY. Verified contract (live): POST /search with
 * `{ query, max_results }` answers `{ results: [{ title, url, content, score }] }`.
 */

const ENDPOINT = 'https://api.tavily.com/search';
const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_RESULTS_LIMIT = 20;

/**
 * Searches the web and returns trimmed, URL-bearing results.
 *
 * @param {string} query
 * @param {{ maxResults?: number, timeoutMs?: number, signal?: AbortSignal }} [options]
 * @returns {Promise<Array<{ title: string, url: string, snippet: string }>>}
 */
export async function searchWeb(query, options = {}) {
  const { maxResults = 5, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = options;

  if (!query || !query.trim()) {
    throw new AppError('The search query was empty.', 422);
  }

  const apiKey = process.env.TAVILY_API_KEY?.trim();
  if (!apiKey) {
    throw new AppError(
      'Web search is not configured: set TAVILY_API_KEY in .env (free key at tavily.com) so research steps get live results.',
      503,
    );
  }

  let response;
  try {
    response = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        query: query.trim(),
        max_results: Math.min(Math.max(maxResults, 1), MAX_RESULTS_LIMIT),
      }),
      signal: signal ?? AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    throw new AppError(`The search provider could not be reached: ${getErrorMessage(error)}`, 502);
  }

  if (!response.ok) {
    let detail = '';
    try {
      detail = JSON.stringify(await response.json()).slice(0, 200);
    } catch {
      /* a body that is not JSON says nothing worth keeping */
    }
    if (response.status === 401 || response.status === 403) {
      throw new AppError('Tavily rejected the API key. Check TAVILY_API_KEY in .env, or generate a new one at app.tavily.com.', 401);
    }
    if (response.status === 429) {
      const retryAfter = Number(response.headers.get('retry-after'));
      const wait = Number.isFinite(retryAfter) && retryAfter > 0 ? ` Retry in about ${Math.ceil(retryAfter / 60)} min.` : '';
      throw new AppError(`Tavily rate-limited the search.${wait}`, 429);
    }
    throw new AppError(`Tavily search failed with HTTP ${response.status}. ${detail}`.trim(), 502);
  }

  /** @type {any} */
  let data;
  try {
    data = await response.json();
  } catch {
    throw new AppError('Tavily returned something that is not JSON.', 502);
  }

  const results = (Array.isArray(data?.results) ? data.results : [])
    .filter((entry) => typeof entry?.url === 'string' && entry.url)
    .map((entry) => ({
      title: typeof entry.title === 'string' ? entry.title : '',
      url: entry.url,
      snippet: typeof entry.content === 'string' ? entry.content : '',
    }));

  if (!results.length) {
    throw new AppError(`Tavily returned no results for "${query.trim().slice(0, 80)}".`, 502);
  }

  return results;
}

/**
 * Renders search results as a prompt appendix the model can reason over.
 *
 * Kept next to the fetcher so the wire format and the prompt format stay one
 * decision.
 *
 * @param {string} query
 * @param {Array<{ title: string, url: string, snippet: string }>} results
 * @returns {string}
 */
export function renderSearchResults(query, results) {
  return [
    '',
    '---',
    `Web search results for "${query}" (fetched live via Tavily just before this request):`,
    ...results.map((result, index) => `[${index + 1}] ${result.title}\n    ${result.url}\n    ${result.snippet.trim()}`),
    '',
    'Treat these results as your primary evidence where relevant, and name the source URLs you relied on.',
    'If the results do not cover something the request asks for, say so explicitly instead of inventing it.',
  ].join('\n');
}
