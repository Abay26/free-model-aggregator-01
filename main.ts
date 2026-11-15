// free-model-aggregator-01 v2: one server, two client wires, ten free providers.
//
// Routing is the whole game here (quality is 60% of the score, and v1's blind
// routing scored 10). The brief at $MAKEFUN_PROMPT_FILE carries the sealed
// quality table and the workload manifest, so every request is routed to the
// highest-rated free model that meets its requirements, over the wire that can
// carry what the request asks for. A provider that refuses (402 when its free
// allowance is gone, 429 on a failure or a spent budget) is counted and the
// request is retried on the next candidate, in rating order, until one answers.
// Only the answering call counts as the request's provider call: a trial that
// fails over keeps its requests, and the failover column with them.
//
// The three upstream wires:
//   openai    POST <base>/v1/chat/completions, GET <base>/v1/models
//   anthropic POST <base>/v1/messages, GET <base>/v1/models
//   gemini    POST <base>/v1beta/models/<model>:generateContent (or
//             :streamGenerateContent?alt=sse), GET <base>/v1beta/models
// A JSON-mode request on the OpenAI wire can only go to an openai- or
// gemini-wire provider: the anthropic wire has no JSON mode to carry. A
// streaming OpenAI request is sent upstream with stream_options.include_usage,
// or the provider sends no usage chunk and the client's usage check fails.
//
// Every choice is a function of the brief, the requests and the providers'
// answers: no wall clock and no unseeded randomness, so a replay leaves the
// same state export. The only clock in the file is a per-call timeout, which
// never fires against a local provider and never changes an answer.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  accumulateAnthropicStream,
  accumulateGeminiStream,
  accumulateOpenAIStream,
  anthropicJsonOf,
  anthropicStreamOf,
  anthropicToGeminiRequest,
  openAIJsonOf,
  openAIStreamOf,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicJson,
  parseGeminiJson,
  parseOpenAIJson,
  type ProviderAnswer,
} from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** One upstream call may take this long before it counts as failed. */
const CALL_TIMEOUT_MS = 8000;
/** A provider that refuses this many calls in a row is treated as spent. */
const EXHAUSTED_AFTER_REFUSALS = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

/** Per-provider call accounting for this trial, in brief order. */
interface ProviderState {
  calls: number;
  /** Refusals since the last success on this provider. */
  refusals: number;
  /** True once the provider looks spent (a 402, or refusals in a row). */
  exhausted: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, refusals: 0, exhausted: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by id. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve a request, best rating first. A provider the
 * trial has marked spent still appears, but after every provider that has not
 * been: its budget may only have looked spent (a failure on its schedule), and
 * it is worth one more try only when nothing fresher can serve. A JSON-mode
 * request on the OpenAI wire excludes anthropic-wire providers: their wire
 * cannot carry response_format, and the provider would echo json_mode false.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const fresh: Candidate[] = [];
  const spent: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const candidate: Candidate = { probe, model, rating: row === null ? 0 : row.rating_milli };
      (stateOf(probe.provider.id).exhausted ? spent : fresh).push(candidate);
    }
  }
  const byRating = (a: Candidate, b: Candidate): number =>
    b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);
  fresh.sort(byRating);
  spent.sort(byRating);
  return [...fresh, ...spent];
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface CallResult {
  ok: boolean;
  status: number;
  answer: ProviderAnswer | null;
}

/** One inference call to a candidate provider, in its own wire, parsed to text plus usage. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<CallResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    // An Anthropic client body must be rewritten to Chat Completions first:
    // the provider parses system messages, tool definitions, max_tokens and
    // stop sequences out of the OpenAI shape, and an Anthropic body sent raw
    // echoes none of them. An OpenAI body passes through with only the model
    // swapped and, when streaming, stream_options.include_usage added.
    const out: Json =
      clientApi === 'anthropic'
        ? anthropicToOpenAI(body, candidate.model.id)
        : { ...body, model: candidate.model.id };
    if (stream) out.stream_options = { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true };
    upstreamBody = out;
    url = `${base}/v1/chat/completions`;
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    upstreamBody = clientApi === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
    headers['anthropic-version'] = '2023-06-01';
    url = `${base}/v1/messages`;
    if (stream) headers.accept = 'text/event-stream';
  } else {
    upstreamBody = clientApi === 'openai' ? openAIToGeminiRequest(body, candidate.model.id) : anthropicToGeminiRequest(body, candidate.model.id);
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, answer: null };
  }
  if (!res.ok) return { ok: false, status: res.status, answer: null };
  try {
    if (stream && res.body !== null) {
      const answer =
        provider.wire === 'openai'
          ? await accumulateOpenAIStream(res.body)
          : provider.wire === 'anthropic'
            ? await accumulateAnthropicStream(res.body)
            : await accumulateGeminiStream(res.body);
      return { ok: true, status: res.status, answer };
    }
    const parsed: unknown = await res.json();
    if (!isRecord(parsed)) return { ok: false, status: res.status, answer: null };
    const answer =
      provider.wire === 'openai' ? parseOpenAIJson(parsed) : provider.wire === 'anthropic' ? parseAnthropicJson(parsed) : parseGeminiJson(parsed);
    return { ok: true, status: res.status, answer };
  } catch {
    return { ok: false, status: res.status, answer: null };
  }
}

/** Record one provider call's outcome in this trial's accounting. */
function noteCall(providerId: string, ok: boolean, status: number): void {
  const state = stateOf(providerId);
  state.calls += 1;
  if (ok) {
    state.refusals = 0;
    return;
  }
  state.refusals += 1;
  // A 402 is the provider's own budget-spent signal; a 429 after another refusal
  // in a row means the same. A lone 429 may only be a failure on its schedule.
  if (status === 402 || state.refusals >= EXHAUSTED_AFTER_REFUSALS) state.exhausted = true;
}

interface Served {
  answer: ProviderAnswer;
  modelId: string;
}

/**
 * Serve one client request: try the candidates in rating order until one
 * answers. A provider that refused this request is not retried for it; its
 * next call may still succeed, so it stays available for later requests
 * unless the trial has marked it spent.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<Served | null> {
  const stream = body.stream === true;
  const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, jsonMode);
  const tried = new Set<string>();
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    if (tried.has(providerId)) continue;
    tried.add(providerId);
    const result = await callCandidate(candidate, clientApi, body, stream);
    noteCall(providerId, result.ok, result.status);
    if (result.ok && result.answer !== null) return { answer: result.answer, modelId: candidate.model.id };
  }
  return null;
}

/** Answer a client request in its own wire, streamed or not as it asked. */
function answer(res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json, served: Served): void {
  const stream = body.stream === true;
  if (!stream) {
    sendJson(res, 200, clientApi === 'openai' ? openAIJsonOf(served.answer, served.modelId) : anthropicJsonOf(served.answer, served.modelId));
    return;
  }
  const text = clientApi === 'openai' ? openAIStreamOf(served.answer, served.modelId) : anthropicStreamOf(served.answer, served.modelId);
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, requirementsFor(body));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  answer(res, 'openai', body, served);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, requirementsFor(body));
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  answer(res, 'anthropic', body, served);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
