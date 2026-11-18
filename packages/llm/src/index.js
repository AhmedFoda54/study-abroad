import OpenAI from 'openai';
import { AppError, getErrorMessage } from '@flow/shared';
import { generateDemoOutput } from './demo.js';
import { runOpencodePrompt } from './opencode.js';
import { runGeminiPrompt, geminiModelId } from './gemini.js';

export { generateDemoOutput } from './demo.js';
export { assertGeminiDataPolicy } from './gemini.js';

/**
 * The single place the platform talks to a model.
 *
 * Four engines sit behind one function. The live engine calls the OpenAI
 * Responses API; the demo engine synthesises structurally valid output from the
 * same JSON schema; the dev-only OpenCode engine routes the prompt through the
 * local OpenCode client, whose free models cost nothing; and the dev-only
 * Gemini engine uses Google's free tier through its OpenAI-compatible Chat
 * Completions endpoint. Every screen, pipeline, and export therefore works
 * before an API key exists, which is what makes the product explorable on
 * first run.
 */

/**
 * Whether model calls should ride the local OpenCode client instead of the
 * OpenAI API. Opt-in per deployment through the environment, and meant for
 * local development: hosted mode cannot spawn processes and this path has no
 * web search or guaranteed structured output.
 * @returns {boolean}
 */
export function isOpencodeEngine() {
  return process.env.FLOW_LLM_ENGINE?.trim().toLowerCase() === 'opencode';
}

/**
 * Whether model calls should use Google's free Gemini tier. Dev-only for the
 * same reason as the OpenCode engine: the free tier's quota is small and its
 * prompts may be used to improve Google products.
 * @returns {boolean}
 */
export function isGeminiEngine() {
  return process.env.FLOW_LLM_ENGINE?.trim().toLowerCase() === 'gemini';
}

/**
 * The model label for logs and screens. A step stores an OpenAI model id, so
 * printing it while a dev engine answers the call claims a model that never
 * ran. This resolves the engine actually in use.
 * @param {string} stepModel
 * @returns {string}
 */
export function describeEngineModel(stepModel) {
  if (isGeminiEngine()) return `${geminiModelId()} (gemini)`;
  if (isOpencodeEngine()) return `${process.env.FLOW_OPENCODE_MODEL?.trim() || stepModel} (opencode)`;
  return stepModel;
}

/** @type {OpenAI | null} */
let cachedClient = null;
/** @type {string | null} */
let cachedKey = null;

/**
 * @param {string} apiKey
 * @returns {OpenAI}
 */
function getClient(apiKey) {
  if (!cachedClient || cachedKey !== apiKey) {
    cachedClient = new OpenAI({ apiKey, maxRetries: 0 });
    cachedKey = apiKey;
  }
  return cachedClient;
}

/**
 * Retries only transient failures. Rate limits and 5xx back off; a bad request
 * or a rejected key fails immediately so the user sees the real reason.
 * @template T
 * @param {() => Promise<T>} fn
 * @param {{ retries?: number, onRetry?: (attempt: number, delayMs: number, reason: string) => void }} [options]
 * @returns {Promise<T>}
 */
async function withRetry(fn, options = {}) {
  const { retries = 3, onRetry } = options;
  let lastError;

  for (let attempt = 0; attempt < retries; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const status = /** @type {{ status?: number }} */ (error)?.status;
      const retriable = status === 429 || status === 408 || (typeof status === 'number' && status >= 500) || status === undefined;

      if (!retriable || attempt === retries - 1) throw error;

      const delay = Math.min(1000 * 2 ** attempt, 15_000);
      onRetry?.(attempt + 1, delay, getErrorMessage(error));
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

/**
 * Normalises usage across model families. Reasoning models report their hidden
 * thinking tokens in a nested field, and those bill as output — so they are
 * counted, not dropped.
 * @param {any} response
 * @param {string} model
 */
function readUsage(response, model) {
  const usage = response?.usage ?? {};
  const promptTokens = usage.input_tokens ?? usage.prompt_tokens ?? 0;
  const completionTokens = usage.output_tokens ?? usage.completion_tokens ?? 0;
  const reasoningTokens =
    usage.output_tokens_details?.reasoning_tokens ?? usage.completion_tokens_details?.reasoning_tokens ?? 0;

  return {
    promptTokens,
    // Reasoning tokens are reported inside output tokens; separate them so the
    // UI can explain a large bill instead of showing an unexplained number.
    completionTokens: Math.max(0, completionTokens - reasoningTokens),
    reasoningTokens,
    totalTokens: usage.total_tokens ?? promptTokens + completionTokens,
    model,
  };
}

/**
 * Pulls text out of a Responses API result, tolerating the shape differences
 * between model families.
 * @param {any} response
 * @returns {string}
 */
function readText(response) {
  if (typeof response?.output_text === 'string' && response.output_text.trim()) {
    return response.output_text;
  }

  const chunks = [];
  for (const item of response?.output ?? []) {
    for (const part of item?.content ?? []) {
      if (typeof part?.text === 'string') chunks.push(part.text);
    }
  }
  if (chunks.length) return chunks.join('\n');

  const legacy = response?.choices?.[0]?.message?.content;
  return typeof legacy === 'string' ? legacy : '';
}

/**
 * Strips code fences a model sometimes wraps JSON in, then parses.
 * @param {string} text
 * @returns {unknown}
 */
function parseJsonOutput(text) {
  const cleaned = String(text ?? '')
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/, '')
    .trim();

  if (!cleaned) throw new AppError('The model returned an empty response.', 502);

  try {
    return JSON.parse(cleaned);
  } catch {
    // Last resort: take the outermost JSON object in the text.
    const first = cleaned.indexOf('{');
    const last = cleaned.lastIndexOf('}');
    if (first !== -1 && last > first) {
      try {
        return JSON.parse(cleaned.slice(first, last + 1));
      } catch {
        /* fall through to the error below */
      }
    }
    throw new AppError('The model did not return valid JSON. Try lowering the temperature or simplifying the output schema.', 502);
  }
}

/**
 * Runs one prompt and returns structured output plus usage.
 *
 * @param {{
 *   systemPrompt?: string,
 *   userPrompt: string,
 *   config: { model: string, temperature: number | null, reasoningEffort: string | null, maxOutputTokens: number | null, webSearch: boolean },
 *   outputSchema?: Record<string, unknown> | null,
 *   schemaName?: string,
 *   apiKey?: string | null,
 *   demoMode?: boolean,
 *   demoSeed?: string,
 *   demoHints?: { targetName?: string, organization?: string, applicantName?: string, focus?: string },
 *   signal?: AbortSignal,
 *   logger?: { info: (m: string) => void, warn: (m: string) => void, debug: (m: string) => void }
 * }} request
 * @returns {Promise<{ output: unknown, text: string, usage: ReturnType<typeof readUsage>, engine: 'live' | 'demo' | 'opencode' | 'gemini' }>}
 */
export async function runPrompt(request) {
  const {
    systemPrompt = '',
    userPrompt,
    config,
    outputSchema = null,
    schemaName = 'structured_output',
    apiKey,
    demoMode = false,
    demoSeed = '',
    demoHints = {},
    signal,
    logger,
  } = request;

  if (!userPrompt || !userPrompt.trim()) {
    throw new AppError('The rendered prompt was empty. Check the template for this step.', 422);
  }

  const engine = isGeminiEngine() ? 'gemini' : isOpencodeEngine() ? 'opencode' : 'openai';
  const shouldDemo = demoMode || (engine === 'openai' && !apiKey);

  if (shouldDemo) {
    logger?.debug(`demo engine (${config.model})`);
    const started = Date.now();
    // A short delay keeps the live console readable; without it every step
    // completes in the same millisecond and the progress UI looks broken.
    await new Promise((resolve) => setTimeout(resolve, 320 + Math.random() * 480));

    const output = generateDemoOutput({ outputSchema, systemPrompt, userPrompt, seed: demoSeed, hints: demoHints });
    const text = JSON.stringify(output, null, 2);
    const approxPrompt = Math.round((systemPrompt.length + userPrompt.length) / 4);

    return {
      output,
      text,
      engine: 'demo',
      usage: {
        promptTokens: approxPrompt,
        completionTokens: Math.round(text.length / 4),
        reasoningTokens: 0,
        totalTokens: approxPrompt + Math.round(text.length / 4),
        model: `${config.model} (demo)`,
      },
      durationMs: Date.now() - started,
    };
  }

  if (engine === 'opencode') {
    // With the dev engine the model id must be an OpenCode id; the stored
    // settings default to OpenAI names, so the env override decides.
    const modelId = process.env.FLOW_OPENCODE_MODEL?.trim() || config.model;
    if (config.webSearch) {
      logger?.warn(`"${config.model}" asked for web search, which the opencode engine cannot do; it ran without.`);
    }

    const started = Date.now();
    const { text } = await runOpencodePrompt({ systemPrompt, userPrompt, model: modelId, signal, logger });

    return {
      output: outputSchema ? parseJsonOutput(text) : text,
      text,
      usage: {
        promptTokens: Math.round((systemPrompt.length + userPrompt.length) / 4),
        completionTokens: Math.round(text.length / 4),
        reasoningTokens: 0,
        totalTokens: Math.round((systemPrompt.length + userPrompt.length + text.length) / 4),
        model: `${modelId} (opencode)`,
      },
      engine: 'opencode',
      durationMs: Date.now() - started,
    };
  }

  if (engine === 'gemini') {
    const started = Date.now();
    const { text, usage } = await runGeminiPrompt({
      systemPrompt,
      userPrompt,
      config,
      outputSchema,
      schemaName,
      signal,
      logger,
    });

    return {
      output: outputSchema ? parseJsonOutput(text) : text,
      text,
      usage,
      engine: 'gemini',
      durationMs: Date.now() - started,
    };
  }

  const client = getClient(apiKey);

  /** @type {Record<string, unknown>} */
  const payload = {
    model: config.model,
    input: systemPrompt
      ? [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ]
      : userPrompt,
  };

  // Reasoning models reject `temperature`; only send it when it is set.
  if (typeof config.temperature === 'number') payload.temperature = config.temperature;
  if (config.reasoningEffort) payload.reasoning = { effort: config.reasoningEffort };
  if (config.maxOutputTokens) payload.max_output_tokens = config.maxOutputTokens;
  if (config.webSearch) payload.tools = [{ type: 'web_search_preview' }];

  if (outputSchema) {
    payload.text = {
      format: {
        type: 'json_schema',
        name: schemaName,
        strict: true,
        schema: outputSchema,
      },
    };
  }

  const response = await withRetry(
    () => client.responses.create(payload, { signal }),
    {
      onRetry: (attempt, delay, reason) => logger?.warn(`model call failed (${reason}); retry ${attempt} in ${delay}ms`),
    },
  ).catch((error) => {
    throw translateOpenAiError(error, config.model);
  });

  const text = readText(response);
  const output = outputSchema ? parseJsonOutput(text) : text;

  return { output, text, usage: readUsage(response, config.model), engine: 'live' };
}

/**
 * Turns an SDK error into something an applicant can act on. Raw SDK messages
 * name internal parameters, which sends people to the wrong screen.
 * @param {unknown} error
 * @param {string} model
 * @returns {AppError}
 */
function translateOpenAiError(error, model) {
  const status = /** @type {{ status?: number }} */ (error)?.status;
  const message = getErrorMessage(error);

  if (status === 401) {
    return new AppError('The OpenAI API key was rejected. Add a valid key under Settings.', 401);
  }
  if (status === 403) {
    return new AppError(`Your key cannot reach "${model}". Pick a different model under Settings.`, 403);
  }
  if (status === 404) {
    return new AppError(`Model "${model}" does not exist or is not available to your account.`, 404);
  }
  if (status === 429) {
    const code = /** @type {{ error?: { code?: string, type?: string } }} */ (error)?.error;
    const outOfCredit =
      code?.code === 'credit_balance_exhausted' ||
      code?.type === 'insufficient_quota' ||
      /quota|credit|billing/i.test(message);

    if (outOfCredit) {
      return new AppError(
        'This OpenAI account has no credit left. Add credit at platform.openai.com/settings/organization/billing, then run again.',
        429,
      );
    }

    return new AppError('OpenAI rate-limited this run. Lower the concurrency under run options and try again.', 429);
  }
  if (status === 400 && /temperature/i.test(message)) {
    // Temperature is no longer sent, so reaching this means a call bypassed
    // resolveModelConfig rather than anything the user can fix.
    return new AppError(`Model "${model}" rejected a parameter this app should not be sending. Please report this.`, 400);
  }
  if (status === 400 && /schema|json/i.test(message)) {
    return new AppError(`The output schema was rejected: ${message}`, 400);
  }
  if (typeof status === 'number' && status >= 500) {
    return new AppError('OpenAI is having trouble right now. The run can be retried in a moment.', 502);
  }

  return new AppError(message, status ?? 502);
}
