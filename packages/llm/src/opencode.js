import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { AppError } from '@flow/shared';

/**
 * The dev-only OpenCode engine.
 *
 * Free models are exclusive to the OpenCode client, so a local development run
 * can borrow the client itself: the OpenCode CLI talks to its own background
 * service, which carries the client's credentials — the same path the OpenCode
 * TUI uses. No API key is involved and the free window applies as usual.
 *
 * This is deliberately not available in hosted mode: the browser cannot spawn
 * processes, and this path has no guaranteed structured-output support, no web
 * search, and the free tier's daily caps. It exists so a developer can exercise
 * the whole pipeline for $0 before paying for anything.
 *
 * The endpoint is `POST /api/experimental/generate/text` on the local service,
 * called through the CLI so authentication stays inside OpenCode. Documented
 * success shape: `{ data: { text: string } }`; failures arrive as a JSON error
 * object with `_tag`, `reason`, `status`, and sometimes `retryAfterMs`.
 */

const DEFAULT_TIMEOUT_MS = 300_000;

/**
 * Locates the OpenCode executable. The npm global install path covers the
 * common Windows case where `opencode` is a shim `spawn` cannot run directly.
 * @returns {string}
 */
function resolveOpencodeBin() {
  const override = process.env.FLOW_OPENCODE_BIN?.trim();
  if (override) {
    if (!existsSync(override)) {
      throw new AppError(`FLOW_OPENCODE_BIN points at "${override}", which does not exist.`, 500);
    }
    return override;
  }

  if (process.platform === 'win32') {
    const candidate = path.join(process.env.APPDATA ?? '', 'npm', 'node_modules', '@opencode', 'cli', 'bin', 'opencode.exe');
    if (existsSync(candidate)) return candidate;
    throw new AppError(
      'The OpenCode CLI was not found. Install it (npm i -g @opencode/cli), or set FLOW_OPENCODE_BIN to its executable path.',
      500,
    );
  }

  return 'opencode';
}

/**
 * @param {string[]} args
 * @param {{ timeoutMs?: number, signal?: AbortSignal }} [options]
 * @returns {Promise<{ code: number | null, stdout: string, stderr: string }>}
 */
function runCli(args, options = {}) {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal } = options;

  return new Promise((resolve, reject) => {
    // A run cancelled while the step was starting must not spawn a full call.
    if (signal?.aborted) {
      reject(new AppError('The model call was cancelled.', 499));
      return;
    }

    const child = spawn(resolveOpencodeBin(), args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    let settled = false;

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener('abort', onAbort);
      child.kill();
      reject(new AppError(`The OpenCode engine did not answer within ${Math.round(timeoutMs / 1000)}s.`, 504));
    }, timeoutMs);

    const onAbort = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.kill();
      reject(new AppError('The model call was cancelled.', 499));
    };
    signal?.addEventListener('abort', onAbort, { once: true });

    child.stdout.on('data', (chunk) => (stdout += chunk));
    child.stderr.on('data', (chunk) => (stderr += chunk));
    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      reject(new AppError(`The OpenCode engine could not be started: ${error.message}`, 500));
    });
    child.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      resolve({ code, stdout, stderr });
    });
  });
}

/**
 * Turns a failed CLI call into an actionable error. The shapes below are the
 * ones the service actually returns (verified live): a provider rejection
 * carries `_tag`, `reason`, `status`, and `retryAfterMs`; a transport failure
 * may only reach stderr.
 * @param {string} stdout
 * @param {string} stderr
 * @returns {AppError}
 */
function translateOpencodeError(stdout, stderr) {
  /** @type {any} */
  let parsed = null;
  try {
    parsed = JSON.parse(stdout);
  } catch {
    /* fall through to the generic failure */
  }

  const status = typeof parsed?.status === 'number' ? parsed.status : undefined;
  const reason = typeof parsed?.reason === 'string' ? parsed.reason : '';
  const message = typeof parsed?.message === 'string' ? parsed.message : '';
  const combined = `${reason} ${message}`;

  if (parsed?._tag === 'ProviderRequestError' || reason) {
    if (status === 429 || reason === 'rate_limit' || /free usage limit|FreeUsageLimit/i.test(combined)) {
      const hours =
        typeof parsed?.retryAfterMs === 'number' && parsed.retryAfterMs > 0
          ? ` The free window resets in about ${Math.max(1, Math.round(parsed.retryAfterMs / 3_600_000))}h.`
          : '';
      return new AppError(
        `OpenCode's free-model window is used up for today.${hours} Wait for the reset, or set FLOW_LLM_ENGINE=openai with a key to keep working.`,
        429,
      );
    }
    if (status === 401 || reason === 'authentication') {
      return new AppError(
        'The OpenCode engine could not authenticate this model. Free models ride your OpenCode client, so pick a free model with FLOW_OPENCODE_MODEL (for example glm-5.3-flash-free).',
        401,
      );
    }
    if (/insufficient|balance|credit/i.test(combined)) {
      return new AppError(
        'This model is billed to OpenCode credits and the balance is empty. Top up at opencode.ai/console, or pick a free model with FLOW_OPENCODE_MODEL.',
        402,
      );
    }
    return new AppError(`The OpenCode engine failed: ${message || reason || 'unknown provider error'}`, status ?? 502);
  }

  // Observed live when the model reference does not resolve: the client rejects
  // the request before any provider is contacted, which would otherwise read as
  // an API-key problem ("Missing key") when it is really a wrong model id.
  if (parsed?._tag === 'InvalidRequestError') {
    return new AppError(
      `The OpenCode engine rejected the request: ${message || 'invalid request'}. Usually FLOW_OPENCODE_MODEL is not a valid OpenCode model id — list the ids with "opencode api get /api/model".`,
      typeof status === 'number' && status >= 400 && status < 500 ? status : 400,
    );
  }

  const detail =
    message || stderr.trim().split('\n')[0] || stdout.trim().slice(0, 160) || 'no error details';
  return new AppError(`The OpenCode engine call failed: ${detail}`.slice(0, 400), status ?? 502);
}

/**
 * Runs one prompt through the local OpenCode client.
 *
 * The endpoint takes a single prompt, so the system prompt is folded in ahead
 * of the user prompt with a marker. Structured output is not guaranteed by the
 * endpoint — callers parse the text with the same tolerant JSON reader the
 * OpenAI fallback path uses.
 *
 * @param {{
 *   systemPrompt?: string,
 *   userPrompt: string,
 *   model?: string,
 *   timeoutMs?: number,
 *   signal?: AbortSignal,
 *   logger?: { warn: (m: string) => void, debug: (m: string) => void }
 * }} request
 * @returns {Promise<{ text: string }>}
 */
export async function runOpencodePrompt(request) {
  const { systemPrompt = '', userPrompt, model, timeoutMs, signal, logger } = request;

  if (!userPrompt || !userPrompt.trim()) {
    throw new AppError('The rendered prompt was empty. Check the template for this step.', 422);
  }

  const prompt = systemPrompt.trim()
    ? `Instructions — follow them exactly:\n${systemPrompt.trim()}\n\nRequest:\n${userPrompt}`
    : userPrompt;

  /** @type {Record<string, unknown>} */
  const body = { prompt };
  // The configured model id is passed through as-is; with the dev engine the
  // model ids are OpenCode ids (set FLOW_OPENCODE_MODEL in .env).
  if (model) body.model = { providerID: 'opencode', id: model };

  logger?.debug(`opencode engine -> ${model ?? '(client default)'}`);

  const { code, stdout, stderr } = await runCli(
    ['api', 'post', '/api/experimental/generate/text', '--data', JSON.stringify(body)],
    { timeoutMs, signal },
  );

  if (code !== 0) throw translateOpencodeError(stdout, stderr);

  let parsed;
  try {
    parsed = JSON.parse(stdout);
  } catch {
    throw new AppError('The OpenCode engine returned something that is not JSON. Check the OpenCode service logs.', 502);
  }

  const text = typeof parsed?.data?.text === 'string' ? parsed.data.text : typeof parsed?.text === 'string' ? parsed.text : null;
  if (text === null) {
    logger?.warn(`OpenCode returned an unexpected success shape: ${stdout.slice(0, 200)}`);
    throw new AppError('The OpenCode engine answered without any text. Check the OpenCode service logs.', 502);
  }

  return { text };
}
