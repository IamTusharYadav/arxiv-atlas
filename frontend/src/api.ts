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

export interface JobStatus {
  job_id: string;
  status: string; // pending | running | done | error
  progress: ProgressStep[];
  result: QueryResult | null;
  error: string | null;
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

export type PostOutcome =
  | { kind: "done"; result: QueryResult } // semantic cache hit answers inline
  | { kind: "accepted"; jobId: string }; // enqueued; poll getJob until terminal

export async function postQuery(question: string): Promise<PostOutcome> {
  const resp = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (resp.status === 200) return { kind: "done", result: (await resp.json()) as QueryResult };
  if (resp.status === 202) {
    const body = (await resp.json()) as { job_id: string };
    return { kind: "accepted", jobId: body.job_id };
  }
  const retryAfter = Number(resp.headers.get("retry-after"));
  throw new ApiError(await detail(resp), resp.status, isNaN(retryAfter) ? undefined : retryAfter);
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const resp = await fetch(`${API_BASE}/api/query/${jobId}`);
  if (!resp.ok) throw new ApiError(await detail(resp), resp.status);
  return (await resp.json()) as JobStatus;
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
