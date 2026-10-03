"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Chunk = {
  chunk_id: string;
  doc_id: string;
  page: number | null;
  x: number;
  y: number;
  z: number;
};

type Space = {
  method: string;
  dimensions: number;
  corpus_version: string;
  chunks_total: number;
  variance_explained: number[];
  documents: { doc_id: string; title: string }[];
  chunks: Chunk[];
};

type QueryResult = {
  query: { x: number; y: number; z: number };
  embed_input_tokens: number;
  embed_model: string | null;
  top_k: number;
  ranked: { chunk_id: string; score: number }[];
};

type Citation = { doc_title: string; page: number | null; clause: string | null; quote: string };

type AskResponse = {
  answer: string;
  citations: Citation[];
  refused: boolean;
  route: string;
  config_hash: string;
  latency_ms: number;
};

type Camera = { yaw: number; pitch: number; zoom: number };

const HUE_STEP = 137.508; // golden angle: maximally spread hues across ~20 documents

function docColor(docId: string, docIndex: Map<string, number>): string {
  const index = docIndex.get(docId) ?? 0;
  return `hsl(${(index * HUE_STEP) % 360} 72% 62%)`;
}

/** Interpolate slate (far) → amber (near) by retrieval score. */
function scoreColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const r = Math.round(30 + clamped * (253 - 30));
  const g = Math.round(41 + clamped * (224 - 41));
  const b = Math.round(59 + clamped * (71 - 59));
  return `rgb(${r},${g},${b})`;
}

export default function VectorsPage() {
  const [space, setSpace] = useState<Space | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [overlay, setOverlay] = useState<QueryResult | null>(null);
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const camera = useRef<Camera>({ yaw: 0.7, pitch: -0.35, zoom: 1 });
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  const hover = useRef<{ index: number; x: number; y: number } | null>(null);
  const overlayRef = useRef<QueryResult | null>(null);
  overlayRef.current = overlay;

  const docIndex = useMemo(() => {
    const map = new Map<string, number>();
    space?.documents.forEach((doc, index) => map.set(doc.doc_id, index));
    return map;
  }, [space]);

  const scoreByChunk = useMemo(() => {
    const map = new Map<string, number>();
    overlay?.ranked.forEach((entry) => map.set(entry.chunk_id, entry.score));
    return map;
  }, [overlay]);

  const chunkById = useMemo(() => {
    const map = new Map<string, Chunk>();
    space?.chunks.forEach((chunk) => map.set(chunk.chunk_id, chunk));
    return map;
  }, [space]);

  /** Center + extent so the cloud fills the canvas regardless of PCA scale. */
  const frame = useMemo(() => {
    if (!space || space.chunks.length === 0) return null;
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (const chunk of space.chunks) {
      sx += chunk.x;
      sy += chunk.y;
      sz += chunk.z;
    }
    const n = space.chunks.length;
    const center = { x: sx / n, y: sy / n, z: sz / n };
    let maxDist = 0;
    for (const chunk of space.chunks) {
      const d = Math.hypot(chunk.x - center.x, chunk.y - center.y, chunk.z - center.z);
      if (d > maxDist) maxDist = d;
    }
    return { center, extent: maxDist || 1 };
  }, [space]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch(`${API_BASE}/vectors`);
        const body = (await res.json()) as Space | { detail: string };
        if (!res.ok) throw new Error("detail" in body ? body.detail : `HTTP ${res.status}`);
        if (!cancelled) setSpace(body as Space);
      } catch (err) {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : "failed to load");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!space || !frame) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const state = { width: 0, height: 0 };

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      state.width = rect.width;
      state.height = rect.height;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const project = (x: number, y: number, z: number, cam: Camera) => {
      const nx = (x - frame.center.x) / frame.extent;
      const ny = (y - frame.center.y) / frame.extent;
      const nz = (z - frame.center.z) / frame.extent;
      const cosY = Math.cos(cam.yaw);
      const sinY = Math.sin(cam.yaw);
      const x1 = nx * cosY - nz * sinY;
      const z1 = nx * sinY + nz * cosY;
      const cosP = Math.cos(cam.pitch);
      const sinP = Math.sin(cam.pitch);
      const y1 = ny * cosP - z1 * sinP;
      const z2 = ny * sinP + z1 * cosP;
      const perspective = 2.6 / (2.6 + z2);
      const cx = state.width / 2;
      const cy = state.height / 2;
      const scale = Math.min(state.width, state.height) * 0.42 * cam.zoom * perspective;
      return { sx: cx + x1 * scale, sy: cy - y1 * scale, depth: z2 };
    };

    let frameHandle = 0;
    const draw = () => {
      frameHandle = requestAnimationFrame(draw);
      const cam = camera.current;
      const active = overlayRef.current;
      ctx.clearRect(0, 0, state.width, state.height);

      const projected = space.chunks.map((chunk, index) => ({ index, ...project(chunk.x, chunk.y, chunk.z, cam) }));
      const order = projected.map((p) => p.index);
      order.sort((a, b) => projected[a].depth - projected[b].depth);

      const topIds = active ? new Set(active.ranked.slice(0, active.top_k).map((r) => r.chunk_id)) : null;
      const queryPoint = active ? project(active.query.x, active.query.y, active.query.z, cam) : null;

      // Decision lines: query → each retrieved chunk, drawn under the cloud.
      if (active && queryPoint) {
        ctx.strokeStyle = "rgba(244,114,182,0.55)";
        ctx.lineWidth = 1;
        for (const hit of active.ranked.slice(0, active.top_k)) {
          const chunk = chunkById.get(hit.chunk_id);
          if (!chunk) continue;
          const p = project(chunk.x, chunk.y, chunk.z, cam);
          ctx.beginPath();
          ctx.moveTo(queryPoint.sx, queryPoint.sy);
          ctx.lineTo(p.sx, p.sy);
          ctx.stroke();
        }
      }

      const scores = scoreByChunk;
      const values = active ? active.ranked.map((r) => r.score) : [];
      const minScore = values.length ? Math.min(...values) : 0;
      const maxScore = values.length ? Math.max(...values) : 1;
      const span = maxScore - minScore || 1;

      for (const idx of order) {
        const chunk = space.chunks[idx];
        const p = projected[idx];
        const score = scores.get(chunk.chunk_id);
        const isTop = topIds?.has(chunk.chunk_id) ?? false;
        const radius = (isTop ? 4.5 : 2.6) * (0.75 + 0.5 * (p.depth + 1));
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, Math.max(1.2, radius), 0, Math.PI * 2);
        if (score !== undefined) {
          ctx.fillStyle = scoreColor((score - minScore) / span);
        } else {
          ctx.fillStyle = docColor(chunk.doc_id, docIndex);
          ctx.globalAlpha = 0.75;
        }
        ctx.fill();
        ctx.globalAlpha = 1;
        if (isTop) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }

      if (queryPoint) {
        ctx.strokeStyle = "#f472b6";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(queryPoint.sx, queryPoint.sy, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(queryPoint.sx - 12, queryPoint.sy);
        ctx.lineTo(queryPoint.sx + 12, queryPoint.sy);
        ctx.moveTo(queryPoint.sx, queryPoint.sy - 12);
        ctx.lineTo(queryPoint.sx, queryPoint.sy + 12);
        ctx.stroke();
        ctx.fillStyle = "#f472b6";
        ctx.font = "11px ui-monospace, monospace";
        ctx.fillText("query", queryPoint.sx + 14, queryPoint.sy + 4);
      }

      // Tooltip for the hovered point.
      const hovered = hover.current;
      if (hovered) {
        const chunk = space.chunks[hovered.index];
        const score = scoreByChunk.get(chunk.chunk_id);
        const doc = space.documents.find((d) => d.doc_id === chunk.doc_id);
        const lines = [
          doc?.title ?? chunk.doc_id,
          `page ${chunk.page ?? "?"}${score !== undefined ? ` · score ${score.toFixed(4)}` : ""}`,
          chunk.chunk_id,
        ];
        ctx.font = "11px ui-monospace, monospace";
        const boxW = Math.max(...lines.map((line) => ctx.measureText(line).width)) + 16;
        const boxH = lines.length * 15 + 10;
        const bx = Math.min(hovered.x + 12, state.width - boxW - 4);
        const by = Math.min(hovered.y + 12, state.height - boxH - 4);
        ctx.fillStyle = "rgba(15,23,42,0.92)";
        ctx.strokeStyle = "rgba(148,163,184,0.6)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(bx, by, boxW, boxH);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#e2e8f0";
        lines.forEach((line, i) => ctx.fillText(line, bx + 8, by + 18 + i * 15));
      }
    };
    frameHandle = requestAnimationFrame(draw);

    const onPointerDown = (event: PointerEvent) => {
      drag.current = {
        x: event.clientX,
        y: event.clientY,
        yaw: camera.current.yaw,
        pitch: camera.current.pitch,
      };
      canvas.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      if (drag.current) {
        camera.current.yaw = drag.current.yaw + (event.clientX - drag.current.x) * 0.008;
        camera.current.pitch = Math.max(
          -1.5,
          Math.min(1.5, drag.current.pitch + (event.clientY - drag.current.y) * 0.008),
        );
        hover.current = null;
        return;
      }
      // Hit-test: project everything and take the nearest point within 10 px.
      let best: { index: number; dist: number } | null = null;
      space.chunks.forEach((chunk, index) => {
        const p = project(chunk.x, chunk.y, chunk.z, camera.current);
        const dist = Math.hypot(p.sx - mx, p.sy - my);
        if (dist < 10 && (!best || dist < best.dist)) best = { index, dist };
      });
      hover.current = best ? { index: (best as { index: number }).index, x: mx, y: my } : null;
    };
    const onPointerUp = () => {
      drag.current = null;
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      camera.current.zoom = Math.max(0.4, Math.min(5, camera.current.zoom * (1 - event.deltaY * 0.001)));
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      cancelAnimationFrame(frameHandle);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointerleave", onPointerUp);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [space, frame, docIndex, scoreByChunk, chunkById, overlay]);

  async function runQuery() {
    setBusy(true);
    setNote(null);
    try {
      const body = JSON.stringify({ question });
      const [vectorsRes, askRes] = await Promise.all([
        fetch(`${API_BASE}/vectors/query`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body,
        }),
        fetch(`${API_BASE}/ask`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body,
        }),
      ]);
      if (!vectorsRes.ok) {
        const detail = (await vectorsRes.json()) as { detail?: string };
        throw new Error(detail.detail ?? `HTTP ${vectorsRes.status}`);
      }
      setOverlay((await vectorsRes.json()) as QueryResult);
      if (askRes.ok) {
        setAskResult((await askRes.json()) as AskResponse);
      } else {
        setAskResult(null);
        setNote("Retrieval visualised, but /ask failed — no answer to show.");
      }
    } catch (err) {
      setNote(err instanceof Error ? err.message : "request failed");
    } finally {
      setBusy(false);
    }
  }

  const variance = space?.variance_explained ?? [];
  const topHits = overlay
    ? overlay.ranked
        .slice(0, overlay.top_k)
        .map((entry, index) => ({ ...entry, rank: index + 1, chunk: chunkById.get(entry.chunk_id) }))
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Vector space</h1>
        <p className="mt-1 max-w-3xl text-sm text-[var(--muted)]">
          Every stored chunk embedded with <code>gemini-embedding-001</code> and drawn at a PCA
          projection of the embedding space. Ask a question to place the query in the same frame
          and watch what retrieval picked.
        </p>
      </div>

      <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <label className="block text-xs uppercase tracking-wide text-[var(--muted)]" htmlFor="vq">
          Question
        </label>
        <textarea
          id="vq"
          rows={2}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What is the LCR requirement for scheduled commercial banks?"
          className="mt-2 w-full rounded border border-[var(--border)] bg-[var(--background)] p-3 text-sm outline-none focus:border-[var(--accent)]"
        />
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={() => void runQuery()}
            disabled={busy || question.trim().length < 3 || !space}
            className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40"
          >
            {busy ? "Projecting…" : "Project query"}
          </button>
          {note && <p className="text-xs text-amber-300">{note}</p>}
        </div>
      </div>

      {!space && (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-8 text-sm text-[var(--muted)]">
          {loadError ? `Vector space unavailable: ${loadError}` : "Fitting the PCA frame (first load takes a few seconds)…"}
        </div>
      )}

      {space && (
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <div className="space-y-3">
            <div className="relative">
              <canvas
                ref={canvasRef}
                className="h-[480px] w-full cursor-grab rounded-lg border border-[var(--border)] bg-[var(--background)] active:cursor-grabbing"
              />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--muted)]">
              <span>
                {space.chunks_total} chunks · {space.documents.length} documents · drag to rotate,
                scroll to zoom
              </span>
              <span>
                PCA axes capture{" "}
                {variance.map((v) => `${(v * 100).toFixed(1)}%`).join(" / ")} of embedding variance
                — a view of the space, not the decision space
              </span>
            </div>
            <div className="columns-2 gap-6 text-xs text-[var(--muted)]">
              {space.documents.map((doc) => (
                <span
                  key={doc.doc_id}
                  className="mb-1 flex break-inside-avoid items-start gap-1.5"
                >
                  <span
                    className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ background: docColor(doc.doc_id, docIndex) }}
                  />
                  <span>{doc.title}</span>
                </span>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
                Retrieval decision
              </h2>
              {overlay ? (
                <>
                  <p className="mt-2 text-xs text-[var(--muted)]">
                    Cosine similarity in the full stored dimension — the exact order{" "}
                    <code>/ask</code> retrieved. Highlighted points are the top {overlay.top_k}.
                  </p>
                  <ol className="mt-3 space-y-2 text-xs">
                    {topHits.map((hit) => (
                      <li key={hit.chunk_id} className="flex items-start gap-2">
                        <span className="mt-0.5 w-5 shrink-0 text-[var(--muted)]">#{hit.rank}</span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[var(--foreground)]">
                            {space.documents.find((d) => d.doc_id === hit.chunk?.doc_id)?.title ??
                              hit.chunk?.doc_id ??
                              hit.chunk_id}
                          </span>
                          <span className="text-[var(--muted)]">
                            page {hit.chunk?.page ?? "?"} · score {hit.score.toFixed(4)}
                          </span>
                        </span>
                      </li>
                    ))}
                  </ol>
                </>
              ) : (
                <p className="mt-2 text-xs text-[var(--muted)]">
                  Run a query to colour the cloud by similarity and show the top-k hits.
                </p>
              )}
            </section>

            {askResult && (
              <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
                <h2 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
                  Answer
                </h2>
                <p className="mt-2 whitespace-pre-wrap text-sm">{askResult.answer}</p>
                {askResult.citations.length > 0 && (
                  <ul className="mt-3 space-y-2 text-xs text-[var(--muted)]">
                    {askResult.citations.map((c, i) => (
                      <li key={i} className="border-l-2 border-[var(--accent)] pl-3">
                        {c.doc_title}, page {c.page ?? "?"}, {c.clause ?? "n/a"}
                      </li>
                    ))}
                  </ul>
                )}
                <p className="mt-3 font-mono text-xs text-[var(--muted)]">
                  latency={askResult.latency_ms}ms · embed tokens={overlay?.embed_input_tokens ?? 0}
                </p>
              </section>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
