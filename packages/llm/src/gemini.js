import OpenAI from 'openai';
import { AppError, getErrorMessage } from '@flow/shared';

/**
 * The Gemini dev engine.
 *
 * Gemini 4 Hydrogen is on Google's free tier, served through an
 * OpenAI-compatible Chat Completions endpoint — so this adapter reuses the
 * existing `openai` dependency with a different base URL. Unlike the OpenCode
 * engine it needs no local process, and unlike the paid OpenAI path it needs
 * no credit card; the cost is a free key and a small daily request quota.
 *
 * Two deliberate differences from the live engine:
 * - Chat Completions, not the Responses API. Structured output therefore uses
 *   `response_format.json_schema`, and usage is read from Chat Completions
 *   fields.
 * - Thinking is pinned to the lowest level. Gemini 4 defaults to `high`, and
 *   thinking tokens are billed as output, so a dev pipeline run would pay for
 *   reasoning it does not need on every step.
 *
 * Data policy matters here: free-tier content may be used to improve Google
 * products, so the orchestrator refuses non-sample profiles unless
 * FLOW_GEMINI_ALLOW_REAL_DATA=1 is set explicitly.
 */

const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/';
const DEFAULT_MODEL = 'gemini-4-hydrogen';
const DEV_THINKING_EFFORT = 'low';

/**
 * The model id for this deployment. Free-tier model availability changes, so
 * it is an env override rather than a hardcoded constant.
 * @returns {string}
 */
function geminiModelId() {
  return process.env.FLOW_GEMINI_MODEL?.trim() || DEFAULT_MODEL;
}

/**
 * Base URL override exists for the test harness (a local fake server) and for
 * future OpenAI-compatible providers; production defaults to Google.
 * @returns {string}
 */
function geminiBaseUrl() {
  return process.env.FLOW_GEMINI_BASE_URL?.trim() || DEFAULT_BASE_URL;
}

/**
 * Refuses to send a non-sample applicant through the free tier unless the
 * deployment has explicitly opted in. Google may use free-tier prompts to
 * improve its products; the paid OpenAI path and seeded sample data are not
 * affected.
 *
 * The caller decides whether the Gemini engine is active — this function is
 * purely the data policy, which keeps it easy to test directly.
 *
 * @param {{ tags?: string[] }} profile
 * @throws {AppError} 403 when the profile is not a sample and no override is set
 */
export function assertGeminiDataPolicy(profile) {
  if (process.env.FLOW_GEMINI_ALLOW_REAL_DATA === '1') return;
  if ((profile?.tags ?? []).includes('sample')) return;

  throw new AppError(
    'The Gemini free engine may use prompts to improve Google products, so it only runs sample profiles unless FLOW_GEMINI_ALLOW_REAL_DATA=1 is set in .env.',
    403,
  );
}

/**
 * Normalises Chat Completions usage. Gemini reports thinking tokens in two
 * different places depending on the response: sometimes inside
 * `completion_tokens_details.reasoning_tokens` (and therefore already inside
 * `completion_tokens`), and sometimes only in `total_tokens`, leaving the
 * parts short of the total. Both are handled, so the recount always adds up.
 * @param {any} usage
 * @param {string} model
 * @returns {{ promptTokens: number, completionTokens: number, reasoningTokens: number, totalTokens: number, model: string }}
 */
function readChatUsage(usage, model) {
  const promptTokens = Number(usage?.prompt_tokens ?? 0) || 0;
  const completionTotal = Number(usage?.completion_tokens ?? 0) || 0;
  const totalTokens = Number(usage?.total_tokens ?? promptTokens + completionTotal) || 0;
  const reportedReasoning = Number(usage?.completion_tokens_details?.reasoning_tokens ?? 0) || 0;

  // Anything the total counts that the parts do not explain is hidden thinking.
  const impliedReasoning = Math.max(0, totalTokens - promptTokens - completionTotal);
  const reasoningTokens = Math.max(reportedReasoning, impliedReasoning);

  return {
    promptTokens,
    // When the details field says reasoning is part of completion_tokens, take
    // it out so it is not counted twice; otherwise completion stands alone.
    completionTokens: Math.max(0, completionTotal - reportedReasoning),
    reasoningTokens,
    totalTokens,
    model: `${model} (gemini)`,
  };
}

/**
 * Turns an SDK error into something the developer can act on.
 * @param {any} error
 * @param {string} model
 * @returns {AppError}
 */
function translateGeminiError(error, model) {
  const status = typeof error?.status === 'number' ? error.status : undefined;
  const message = getErrorMessage(error);

  if (status === 401) {
    return new AppError(
      'Google rejected the Gemini API key. Check GEMINI_API_KEY in .env, or create a new free key at aistudio.google.com/apikey.',
      401,
    );
  }
  if (status === 403) {
    return new AppError(
      'Google refused this key (403). Check that the Gemini API is enabled for its project and the key is unrestricted, or create a new free key at aistudio.google.com/apikey.',
      403,
    );
  }
  if (status === 429) {
    return new AppError(
      'The Gemini free-tier quota is used up (a few requests per minute and a small daily limit). Check AI Studio for the reset time, or switch FLOW_LLM_ENGINE=opencode.',
      429,
    );
  }
  if (status === 404) {
    return new AppError(
      `Gemini model "${model}" is not available for this key. Set FLOW_GEMINI_MODEL to a current id (for example gemini-4-hydrogen).`,
      404,
    );
  }
  if (status === 400) {
    return new AppError(`Gemini rejected the request: ${message}`, 400);
  }
  return new AppError(`The Gemini engine call failed: ${message}`, status ?? 502);
}

/**
 * Runs one prompt through Gemini's OpenAI-compatible Chat Completions endpoint.
 *
 * @param {{
 *   systemPrompt?: string,
 *   userPrompt: string,
 *   config: { model: string, maxOutputTokens?: number | null },
 *   outputSchema?: Record<string, unknown> | null,
 *   schemaName?: string,
 *   signal?: AbortSignal,
 *   logger?: { warn: (m: string) => void, debug: (m: string) => void }
 * }} request
 * @returns {Promise<{ text: string, usage: ReturnType<typeof readChatUsage> }>}
 */
export async function runGeminiPrompt(request) {
  const { systemPrompt = '', userPrompt, config, outputSchema = null, schemaName = 'structured_output', signal, logger } = request;

  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    throw new AppError(
      'The Gemini dev engine needs a free API key: set GEMINI_API_KEY in .env (create one at aistudio.google.com/apikey).',
      503,
    );
  }
  if (!userPrompt || !userPrompt.trim()) {
    throw new AppError('The rendered prompt was empty. Check the template for this step.', 422);
  }

  const model = geminiModelId();
  const client = new OpenAI({ apiKey, baseURL: geminiBaseUrl(), maxRetries: 1 });

  /** @type {any[]} */
  const messages = [];
  if (systemPrompt.trim()) messages.push({ role: 'system', content: systemPrompt.trim() });
  messages.push({ role: 'user', content: userPrompt });

  /** @type {Record<string, unknown>} */
  const body = { model, messages, reasoning_effort: DEV_THINKING_EFFORT };
  if (config.maxOutputTokens) body.max_tokens = config.maxOutputTokens;
  if (outputSchema) {
    body.response_format = {
      type: 'json_schema',
      json_schema: { name: schemaName, schema: outputSchema, strict: true },
    };
  }

  logger?.debug(`gemini engine -> ${model} (${geminiBaseUrl()})`);

  /** @type {any} */
  let response;
  try {
    response = await client.chat.completions.create(body, { signal });
  } catch (error) {
    throw translateGeminiError(error, model);
  }

  const finishReason = response?.choices?.[0]?.finish_reason ?? 'unknown';
  const text = response?.choices?.[0]?.message?.content;
  if (typeof text !== 'string' || !text.trim()) {
    logger?.warn(`Gemini answered without text (finish_reason=${finishReason}).`);
    throw new AppError('The Gemini engine answered without any text. Check the AI Studio quota and status pages.', 502);
  }

  return { text, usage: readChatUsage(response.usage, model) };
}
