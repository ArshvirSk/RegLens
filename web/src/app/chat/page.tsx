"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Citation = { doc_title: string; page: number | null; clause: string | null; quote: string };
type AskResponse = {
  answer: string;
  citations: Citation[];
  refused: boolean;
  route: string;
  config_hash: string;
  latency_ms: number;
};

export default function ChatPage() {
  const [question, setQuestion] = useState("");
  const [asOf, setAsOf] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function ask() {
    setBusy(true);
    setNote(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          question,
          as_of_date: asOf.length === 10 ? asOf : null,
        }),
      });
      const body = (await res.json()) as AskResponse | { detail: string };
      if (!res.ok) {
        setNote("detail" in body ? body.detail : `HTTP ${res.status}`);
      } else {
        setResult(body as AskResponse);
      }
    } catch (err) {
      setNote(err instanceof Error ? err.message : "request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Ask</h1>

      <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <label className="block text-xs uppercase tracking-wide text-[var(--muted)]" htmlFor="q">
          Question
        </label>
        <textarea
          id="q"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          placeholder="What is the LCR requirement for scheduled commercial banks?"
          className="mt-2 w-full rounded border border-[var(--border)] bg-[var(--background)] p-3 text-sm outline-none focus:border-[var(--accent)]"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <label className="text-xs text-[var(--muted)]" htmlFor="asof">
            As of date
          </label>
          <input
            id="asof"
            type="date"
            value={asOf}
            onChange={(e) => setAsOf(e.target.value)}
            className="rounded border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm"
          />
          <button
            onClick={() => void ask()}
            disabled={busy || question.trim().length < 3}
            className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
          >
            {busy ? "Asking…" : "Ask"}
          </button>
        </div>
        <p className="mt-3 text-xs text-[var(--muted)]">
          Retrieval and generation arrive in Phase 1. Until then <code>/ask</code> returns an
          explicit &quot;not implemented&quot; response rather than a fabricated answer.
        </p>
      </div>

      {note && (
        <p className="rounded border border-amber-700 bg-amber-950/40 p-4 text-sm text-amber-200">
          {note}
        </p>
      )}

      {result && (
        <article className="space-y-4 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
          <p className="whitespace-pre-wrap text-sm">{result.answer}</p>
          {result.citations.length > 0 && (
            <ul className="space-y-2 text-xs text-[var(--muted)]">
              {result.citations.map((c, i) => (
                <li key={i} className="border-l-2 border-[var(--accent)] pl-3">
                  {c.doc_title}, page {c.page ?? "?"}, {c.clause ?? "n/a"}
                </li>
              ))}
            </ul>
          )}
          <p className="font-mono text-xs text-[var(--muted)]">
            route={result.route} latency={result.latency_ms}ms config={result.config_hash.slice(0, 8)}
          </p>
        </article>
      )}

      <p className="text-xs text-[var(--muted)]">
        Informational only. Not legal or investment advice.
      </p>
    </div>
  );
}
