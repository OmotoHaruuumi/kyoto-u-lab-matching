"use client";

import { useState, useEffect } from "react";
import { searchLabs, LabResult } from "./actions";

// ---------------------------------------------------------------------------
// Detail modal
// ---------------------------------------------------------------------------
function DetailModal({ lab, onClose }: { lab: LabResult; onClose: () => void }) {
  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-[#111827] border border-white/10 rounded-3xl w-full max-w-2xl max-h-[85vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 text-slate-400 hover:text-white transition-colors"
          aria-label="閉じる"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="p-8">
          {/* Header */}
          <h2 className="text-2xl font-bold text-white mb-1 pr-8">{lab.name}</h2>
          {lab.name_en && <p className="text-sm text-slate-400 mb-3">{lab.name_en}</p>}
          <div className="flex flex-wrap gap-2 text-xs text-cyan-200/70 mb-6">
            {lab.department && <span>{lab.department}</span>}
            {lab.faculty && <span>• {lab.faculty}</span>}
          </div>

          {/* Description */}
          {lab.description && (
            <div className="mb-6">
              <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2 font-semibold">研究室紹介</h3>
              <p className="text-slate-300 text-sm leading-relaxed">{lab.description}</p>
            </div>
          )}

          {/* Keywords — all of them */}
          {lab.keywords && lab.keywords.length > 0 && (
            <div className="mb-6">
              <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2 font-semibold">キーワード</h3>
              <div className="flex flex-wrap gap-2">
                {lab.keywords.map((kw, idx) => (
                  <span key={idx} className="px-3 py-1 text-xs rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          )}

          <hr className="border-white/5 my-6" />

          {/* Match highlights — all chunks */}
          <div className="mb-6">
            <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-3 font-semibold">マッチ理由</h3>
            <ul className="space-y-3">
              {lab.matched_chunks.map((chunk, idx) => (
                <li key={idx} className="text-xs bg-black/20 p-3 rounded-lg border border-white/5">
                  <span className="inline-block px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 mb-1 text-[10px]">
                    {chunk.source_type}
                  </span>
                  <p className="text-slate-300 italic whitespace-pre-wrap">&quot;{chunk.chunk_text}&quot;</p>
                </li>
              ))}
            </ul>
          </div>

          {/* Link */}
          {lab.lab_url && (
            <a
              href={lab.lab_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center text-sm font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
            >
              研究室サイトへ
              <svg className="ml-1.5 w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skeleton card shown while search is in progress
// ---------------------------------------------------------------------------
function SkeletonCard() {
  return (
    <div className="bg-[#1a2333]/80 backdrop-blur-xl border border-white/10 rounded-3xl p-8 animate-pulse">
      <div className="flex justify-between items-start mb-4">
        <div className="h-7 bg-slate-700/60 rounded-lg w-3/5" />
        <div className="h-5 bg-slate-700/40 rounded-full w-20" />
      </div>
      <div className="h-4 bg-slate-700/40 rounded w-2/5 mb-6" />
      <div className="space-y-2 mb-6">
        <div className="h-3 bg-slate-700/40 rounded w-full" />
        <div className="h-3 bg-slate-700/40 rounded w-11/12" />
        <div className="h-3 bg-slate-700/40 rounded w-4/5" />
      </div>
      <div className="flex gap-2 mb-6">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-6 w-16 bg-slate-700/40 rounded-full" />
        ))}
      </div>
      <hr className="border-white/5 my-4" />
      <div className="space-y-3">
        <div className="h-3 bg-slate-700/30 rounded w-1/4 mb-2" />
        <div className="h-12 bg-slate-700/30 rounded-lg w-full" />
        <div className="h-12 bg-slate-700/30 rounded-lg w-full" />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state shown when search returns zero results
// ---------------------------------------------------------------------------
const EXAMPLE_QUERIES = [
  "自然言語処理・大規模言語モデル",
  "再生可能エネルギーの最適化",
  "ロボットの自律制御",
  "医療画像診断 AI",
];

function EmptyState({ query }: { query: string }) {
  return (
    <div className="p-12 text-center rounded-3xl border border-white/5 bg-[#1a2333]/50 backdrop-blur-md">
      <div className="text-5xl mb-5">🔬</div>
      <h3 className="text-xl font-semibold text-slate-200 mb-2">
        「{query}」に一致する研究室が見つかりませんでした
      </h3>
      <p className="text-slate-400 mb-6">
        キーワードを変えるか、以下の例を試してみてください。
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        {EXAMPLE_QUERIES.map((q) => (
          <span
            key={q}
            className="px-4 py-2 text-sm rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
          >
            {q}
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------
export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<LabResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [lastQuery, setLastQuery] = useState("");
  const [selectedLab, setSelectedLab] = useState<LabResult | null>(null);

  const handleSearch = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setHasSearched(true);
    setLastQuery(query.trim());

    try {
      const resp = await searchLabs(query);
      if ("error" in resp) {
        setError(resp.error as string);
        setResults([]);
      } else {
        setResults(resp.results || []);
      }
    } catch {
      setError("An unexpected error occurred.");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0f1c] text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Detail modal */}
      {selectedLab && (
        <DetailModal lab={selectedLab} onClose={() => setSelectedLab(null)} />
      )}

      {/* Dynamic Background */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-indigo-900/40 blur-[120px]" />
        <div className="absolute top-[60%] -right-[10%] w-[40%] h-[50%] rounded-full bg-blue-900/30 blur-[120px]" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-16 flex flex-col items-center">
        {/* Header */}
        <div className={`transition-all duration-700 ease-in-out flex flex-col items-center w-full ${hasSearched ? "mt-8" : "mt-32"}`}>
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-300 mb-6 text-center">
            Kyoto Univ. Lab Match
          </h1>
          <p className="text-slate-400 text-lg md:text-xl text-center max-w-2xl mb-12">
            研究課題や興味を自然言語で入力すると、関連する京都大学の研究室を探せます。
          </p>

          <form onSubmit={handleSearch} className="w-full max-w-3xl relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-500" />
            <div className="relative flex items-center bg-[#111827] rounded-2xl overflow-hidden shadow-2xl ring-1 ring-white/10">
              <input
                type="text"
                className="w-full bg-transparent text-white px-8 py-5 outline-none placeholder:text-slate-500 text-lg"
                placeholder="例: 自然言語処理を使った医療診断の研究..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <button
                type="submit"
                disabled={isLoading}
                className="px-8 py-5 bg-indigo-600 hover:bg-indigo-500 transition-colors text-white font-medium text-lg disabled:opacity-50 flex items-center gap-2 whitespace-nowrap"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    検索中…
                  </>
                ) : (
                  "検索"
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Results Section */}
        <div className="mt-16 w-full max-w-5xl">
          {/* Error */}
          {error && (
            <div className="p-6 rounded-2xl bg-red-900/30 border border-red-500/30 text-red-200 text-center backdrop-blur-md mb-6">
              <p>{error}</p>
            </div>
          )}

          {/* Skeleton while loading */}
          {isLoading && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[...Array(4)].map((_, i) => <SkeletonCard key={i} />)}
            </div>
          )}

          {/* Zero results guide */}
          {!isLoading && hasSearched && !error && results.length === 0 && (
            <EmptyState query={lastQuery} />
          )}

          {/* Results grid */}
          {!isLoading && results.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {results.map((lab) => (
                <div
                  key={lab.lab_id}
                  className="group relative bg-[#1a2333]/80 backdrop-blur-xl border border-white/10 rounded-3xl p-8 hover:border-indigo-400/50 transition-all duration-300 hover:shadow-[0_0_30px_-5px_rgba(99,102,241,0.3)] hover:-translate-y-1"
                >
                  <div className="flex flex-col h-full">
                    <div className="mb-4">
                      <div className="flex justify-between items-start mb-2">
                        <h2 className="text-2xl font-bold text-white group-hover:text-indigo-300 transition-colors">
                          {lab.name}
                        </h2>
                        <span className="text-xs font-mono font-medium px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 shrink-0 ml-2">
                          Score: {lab.total_score.toFixed(3)}
                        </span>
                      </div>
                      {lab.name_en && (
                        <p className="text-sm text-slate-400 mb-2">{lab.name_en}</p>
                      )}
                      <div className="flex flex-wrap gap-2 text-xs text-cyan-200/70 mb-4">
                        {lab.department && <span>{lab.department}</span>}
                        {lab.faculty && <span>• {lab.faculty}</span>}
                      </div>
                    </div>

                    <div className="text-slate-300 text-sm leading-relaxed mb-6 line-clamp-3 flex-grow">
                      {lab.description}
                    </div>

                    {/* Keywords (preview) */}
                    {lab.keywords && lab.keywords.length > 0 && (
                      <div className="flex flex-wrap gap-2 mb-6">
                        {lab.keywords.slice(0, 4).map((kw, idx) => (
                          <span key={idx} className="px-3 py-1 text-xs rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                            {kw}
                          </span>
                        ))}
                        {lab.keywords.length > 4 && (
                          <span className="px-3 py-1 text-xs rounded-full bg-slate-800/50 text-slate-400 border border-slate-700/50">
                            +{lab.keywords.length - 4}
                          </span>
                        )}
                      </div>
                    )}

                    <hr className="border-white/5 my-4" />

                    {/* Match highlights (preview) */}
                    <div className="mt-2">
                      <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-3 font-semibold">マッチ理由</h3>
                      <ul className="space-y-3">
                        {lab.matched_chunks.slice(0, 2).map((chunk, idx) => (
                          <li key={idx} className="text-xs text-slate-400 bg-black/20 p-3 rounded-lg border border-white/5">
                            <span className="inline-block px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 mb-1 text-[10px]">
                              {chunk.source_type}
                            </span>
                            <p className="line-clamp-2 italic text-slate-300">&quot;{chunk.chunk_text}&quot;</p>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Footer */}
                    <div className="mt-8 flex items-center justify-between">
                      {lab.lab_url ? (
                        <a
                          href={lab.lab_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center text-sm font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
                        >
                          研究室サイトへ
                          <svg className="ml-1.5 w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                        </a>
                      ) : <span />}
                      <button
                        onClick={() => setSelectedLab(lab)}
                        className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors font-medium"
                      >
                        詳細を見る →
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
