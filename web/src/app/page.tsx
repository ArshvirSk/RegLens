"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Health = {
  status: string;
  version: string;
  env: string;
  corpus_version: string;
  experiment: string;
  config_hash: string;
  database: string;
};

export default function StatusPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setHealth((await res.json()) as Health);
      setError(null);
    } catch (err) {
      setHealth(null);
      setError(err instanceof Error ? err.message : "unknown error");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-semibold">RegLens</h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">
          A cited, temporally-aware question answering system over RBI/SEBI regulation and Indian
          bank filings. Built in phases; every capability is measured on a golden eval set before it
          is kept.
        </p>
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Build status
        </h2>
        <ul className="mt-3 space-y-2 text-sm">
          <li>Phase 0 — foundations, corpus manifest, tracing skeleton: complete</li>
          <li>Phase 1 — naive baseline (fixed chunks, dense retrieval) and first eval: next</li>
          <li>Phase 2 — retrieval quality ablations: not started</li>
          <li>Phase 3 — tables, temporal filters, routing: not started</li>
          <li>Phase 4 — citation verifier, refusals, source viewer: not started</li>
        </ul>
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            API health
          </h2>
          <button
            onClick={() => void load()}
            className="rounded border border-[var(--border)] px-3 py-1 text-xs hover:border-[var(--accent)]"
          >
            Refresh
          </button>
        </div>
        {error && (
          <p className="mt-3 text-sm text-amber-300">
            API unreachable ({error}). Start the stack with <code>make up</code>.
          </p>
        )}
        {health && (
          <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            {Object.entries(health).map(([key, value]) => (
              <div key={key}>
                <dt className="text-xs uppercase text-[var(--muted)]">{key}</dt>
                <dd className="truncate font-mono text-xs">{String(value)}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <p className="text-xs text-[var(--muted)]">
        Raw source PDFs are stored immutably and hashed; only public documents are collected, after
        checking each site&apos;s robots.txt and terms of use.
      </p>
    </div>
  );
}
