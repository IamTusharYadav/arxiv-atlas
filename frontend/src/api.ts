// Typed client for the query API. The shapes mirror the pydantic response models in
// src/atlas_api/app.py; if a field changes there, change it here.

const API_BASE: string = import.meta.env.VITE_API_URL ?? "";

export interface PaperOut {
  arxiv_id: string;
  title: string;
  primary_category: string;
  score: number;
}

export interface TraceStep {
  step: string;
  summary: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface QueryResult {
  brief: string;
  papers: PaperOut[];
  // Semantic-similarity edges among the cited papers only, for the citation graph. Optional so
  // older cached results and the saved demo runs (which predate the field) still type-check.
  links?: GraphLinkOut[];
  trace: TraceStep[];
  cost_usd: number;
  cached: boolean;
  // A cap stopped the run: the brief is gathered evidence, not a synthesized answer.
  partial: boolean;
}

export interface ProgressStep {
  step: string;
  summary: string;
}

export interface JobStatus<T> {
  job_id: string;
  status: string; // pending | running | done | error
  progress: ProgressStep[];
  result: T | null;
  error: string | null;
}

export interface LandscapePaper {
  arxiv_id: string;
  title: string;
  primary_category: string;
  published_month: string; // "YYYY-MM"
}

export interface Direction {
  name: string;
  problem: string;
  papers: LandscapePaper[]; // most-central-first
  representative_ids: string[];
}

export interface TimelinePoint {
  month: string;
  direction: string;
  count: number;
}

export interface ReadingStep {
  arxiv_id: string;
  title: string;
  reason: string;
}

export interface LandscapeResult {
  topic: string;
  overview: string;
  key_ideas: string[];
  directions: Direction[];
  timeline: TimelinePoint[];
  reading_order: ReadingStep[];
  open_problems: string[];
  // Semantic-similarity edges between the landscape's own papers, for the topic map.
  links: GraphLinkOut[];
  trace: TraceStep[];
  cost_usd: number;
  cached: boolean;
  // The pipeline stopped before mapping (out of scope or too few papers);
  // the overview carries the explanation and every list is empty.
  declined: boolean;
}

export interface CorpusStatus {
  status: string;
  corpus_size: number;
}

export interface GraphNodeOut {
  arxiv_id: string;
  title: string;
  primary_category: string;
}

export interface GraphLinkOut {
  source: string;
  target: string;
  weight: number;
}

export interface GraphResponse {
  center: string;
  nodes: GraphNodeOut[];
  links: GraphLinkOut[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfterS?: number,
  ) {
    super(message);
  }
}

async function detail(resp: Response): Promise<string> {
  try {
    const body: unknown = await resp.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const d = (body as { detail: unknown }).detail;
      if (typeof d === "string") return d;
    }
  } catch {
    // fall through to the status line
  }
  return `${resp.status} ${resp.statusText}`;
}

export type PostOutcome<T> =
  | { kind: "done"; result: T } // semantic cache hit answers inline
  | { kind: "accepted"; jobId: string }; // enqueued; poll the job until terminal

async function post<T>(path: string, body: object): Promise<PostOutcome<T>> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (resp.status === 200) return { kind: "done", result: (await resp.json()) as T };
  if (resp.status === 202) {
    const accepted = (await resp.json()) as { job_id: string };
    return { kind: "accepted", jobId: accepted.job_id };
  }
  const retryAfter = Number(resp.headers.get("retry-after"));
  throw new ApiError(await detail(resp), resp.status, isNaN(retryAfter) ? undefined : retryAfter);
}

export function postQuery(question: string): Promise<PostOutcome<QueryResult>> {
  return post("/api/query", { question });
}

export function postLandscape(topic: string): Promise<PostOutcome<LandscapeResult>> {
  return post("/api/landscape", { topic });
}

async function job<T>(path: string): Promise<JobStatus<T>> {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) throw new ApiError(await detail(resp), resp.status);
  return (await resp.json()) as JobStatus<T>;
}

export function getJob(jobId: string): Promise<JobStatus<QueryResult>> {
  return job(`/api/query/${jobId}`);
}

export function getLandscapeJob(jobId: string): Promise<JobStatus<LandscapeResult>> {
  return job(`/api/landscape/${jobId}`);
}

export async function getStatus(): Promise<CorpusStatus> {
  const resp = await fetch(`${API_BASE}/api/status`);
  if (!resp.ok) throw new ApiError(await detail(resp), resp.status);
  return (await resp.json()) as CorpusStatus;
}

export async function getGraph(arxivId: string): Promise<GraphResponse> {
  const resp = await fetch(`${API_BASE}/api/graph/${encodeURIComponent(arxivId)}`);
  if (!resp.ok) throw new ApiError(await detail(resp), resp.status);
  return (await resp.json()) as GraphResponse;
}
