import { createService, finaliseService, listen, route } from '@flow/service-kit';
import { ok, AppError, getErrorMessage, renderTemplate, extractVariables, resolveModelConfig } from '@flow/shared';
import { searchWeb, renderSearchResults } from '@flow/search';
import { runPrompt, isOpencodeEngine } from './index.js';
import { resolveApiKey } from './secrets.js';

/**
 * Builds a stage service.
 *
 * Research, analysis, and outreach are separate deployables with separate
 * scaling characteristics — research is slow and web-bound, outreach is fast
 * and token-heavy — but they execute prompts identically. This factory holds
 * the shared behaviour so the three service files stay declarative.
 *
 * @param {{
 *   name: 'research' | 'analysis' | 'outreach',
 *   port: number | string,
 *   defaultModel: string,
 *   description: string
 * }} definition
 */
export function createExecutorService({ name, port, defaultModel, description }) {
  const { app, logger } = createService({ name });

  app.get('/describe', (_req, res) => {
    res.json(ok({ service: name, description, defaultModel }));
  });

  /**
   * Executes one pipeline step for one target.
   *
   * The whole request is self-contained: the orchestrator sends the prompt
   * text, the config, and the context. Nothing here reads the database, which
   * is what lets these three services scale independently of the rest.
   */
  app.post(
    '/execute',
    route(async (req, res) => {
      const started = Date.now();
      const { step, context = {}, runtime = {} } = req.body ?? {};

      if (!step || typeof step !== 'object') {
        throw new AppError('The request is missing the step definition.', 422);
      }
      if (!step.userPrompt || !String(step.userPrompt).trim()) {
        throw new AppError(`Step "${step.title ?? step.key}" has an empty prompt. Open it in the Prompt studio and add one.`, 422);
      }

      // Decided from code, not from the request: a stored prompt from an older
      // workspace must not be able to send a rejected parameter.
      const config = resolveModelConfig(step.config);

      const system = renderTemplate(step.systemPrompt ?? '', context);
      const user = renderTemplate(step.userPrompt, context);
      const missingVariables = [...new Set([...system.missing, ...user.missing])];

      const apiKey = await resolveApiKey();
      // The OpenCode dev engine carries its own credential (the local client),
      // so a missing OpenAI key does not force demo mode there.
      const demoMode = Boolean(runtime.demoMode) || (!isOpencodeEngine() && !apiKey);

      logger.info(
        `${step.key} -> ${config.model}${config.webSearch ? ' +web' : ''}${demoMode ? ' (demo)' : ''}` +
          `${missingVariables.length ? ` [${missingVariables.length} empty variables]` : ''}`,
      );

      // Search is done by Flow, not by the model: the OpenCode engine's
      // one-shot endpoint cannot run provider tools, so live results go into
      // the prompt as evidence. Only the dev engine needs this — the OpenAI
      // path keeps its native web search — and demo output never searches.
      let searched = false;
      if (config.webSearch && isOpencodeEngine() && !demoMode) {
        const searchQuery = [context?.target?.name, context?.target?.organization].filter(Boolean).join(' ').trim();
        if (searchQuery) {
          try {
            const results = await searchWeb(searchQuery, { maxResults: 5 });
            user.text += renderSearchResults(searchQuery, results);
            searched = true;
            logger.info(`${step.key} +search: "${searchQuery}" -> ${results.length} results`);
          } catch (error) {
            logger.warn(`${step.key} web search failed (${getErrorMessage(error)}); continuing without search.`);
          }
        } else {
          logger.warn(`${step.key} asked for web search, but no target name was available to build a query; continuing without.`);
        }
      }

      const result = await runPrompt({
        systemPrompt: system.text,
        userPrompt: user.text,
        // The executor satisfied the search requirement itself, so the model
        // call must not log a misleading "ran without search" warning.
        config: searched ? { ...config, webSearch: false } : config,
        outputSchema: step.outputSchema ?? null,
        schemaName: String(step.key ?? name).replace(/[^a-zA-Z0-9_]/g, '_').slice(0, 60) || 'structured_output',
        apiKey,
        demoMode,
        demoSeed: `${context?.target?.id ?? ''}:${step.key ?? ''}`,
        // Explicit hints beat parsing them back out of the rendered prompt,
        // where the applicant block and the target block both contain "Name:".
        demoHints: {
          targetName: context?.target?.name ?? '',
          organization: context?.target?.organization ?? '',
          applicantName: context?.applicant?.fullName ?? context?.applicant?.name ?? '',
          focus: context?.target?.focusAreas || context?.applicant?.researchInterests || '',
        },
        logger,
      });

      res.json(
        ok({
          stepKey: step.key,
          output: result.output,
          rawText: result.text,
          usage: result.usage,
          engine: result.engine,
          missingVariables,
          detectedVariables: extractVariables(step.systemPrompt, step.userPrompt),
          renderedPrompt: {
            system: system.text,
            user: user.text,
            approximateTokens: Math.round((system.text.length + user.text.length) / 4),
          },
          durationMs: Date.now() - started,
        }),
      );
    }),
  );

  finaliseService(app, logger);
  listen(app, port, logger);

  return app;
}
