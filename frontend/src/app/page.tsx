"use client";

import { useState } from "react";
import { searchLabs, LabResult } from "./actions";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<LabResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const resp = await searchLabs(query);
      if ("error" in resp) {
        setError(resp.error as string);
        setResults([]);
      } else {
        setResults(resp.results || []);
      }
    } catch (err: any) {
      setError("An unexpected error occurred.");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0f1c] text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Dynamic Background */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-indigo-900/40 blur-[120px]" />
        <div className="absolute top-[60%] -right-[10%] w-[40%] h-[50%] rounded-full bg-blue-900/30 blur-[120px]" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-16 flex flex-col items-center">
        {/* Header Element */}
        <div className={`transition-all duration-700 ease-in-out flex flex-col items-center w-full ${hasSearched ? "mt-8" : "mt-32"}`}>
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-300 mb-6 text-center">
            Kyoto Univ. Lab Match
          </h1>
          <p className="text-slate-400 text-lg md:text-xl text-center max-w-2xl mb-12">
            Describe your research interests in natural language and discover the perfect lab to advance your academic journey.
          </p>

          <form onSubmit={handleSearch} className="w-full max-w-3xl relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-500"></div>
            <div className="relative flex items-center bg-[#111827] rounded-2xl overflow-hidden shadow-2xl ring-1 ring-white/10">
              <input
                type="text"
                className="w-full bg-transparent text-white px-8 py-5 outline-none placeholder:text-slate-500 text-lg"
                placeholder="Ex: Artificial Intelligence in Healthcare..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <button
                type="submit"
                disabled={isLoading}
                className="px-8 py-5 bg-indigo-600 hover:bg-indigo-500 transition-colors text-white font-medium text-lg disabled:opacity-50 flex items-center"
              >
                {isLoading ? (
                  <svg className="animate-spin h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  "Search"
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Results Section */}
        <div className="mt-16 w-full max-w-5xl">
          {error && (
            <div className="p-6 rounded-2xl bg-red-900/30 border border-red-500/30 text-red-200 text-center backdrop-blur-md">
              <p>{error}</p>
            </div>
          )}

          {!isLoading && hasSearched && !error && results.length === 0 && (
            <div className="p-12 text-center text-slate-400 text-lg glassmorphism rounded-3xl border border-white/5">
              No specific labs found for this query. Try different keywords!
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10 w-full">
            {!isLoading && results.map((lab) => (
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
                      <span className="text-xs font-mono font-medium px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                        Score: {lab.total_score.toFixed(3)}
                      </span>
                    </div>
                    {lab.name_en && (
                      <p className="text-sm text-slate-400 mb-2">{lab.name_en}</p>
                    )}
                    <div className="flex flex-wrap gap-2 text-xs text-cyan-200/70 mb-4">
                      {lab.department && (<span>{lab.department}</span>)}
                      {lab.faculty && (<span>• {lab.faculty}</span>)}
                    </div>
                  </div>

                  <div className="text-slate-300 text-sm leading-relaxed mb-6 line-clamp-3 flex-grow">
                    {lab.description}
                  </div>

                  {/* Keywords */}
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

                  {/* Match highlights */}
                  <div className="mt-2">
                    <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-3 font-semibold">Match Reasons</h3>
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

                  {lab.lab_url && (
                    <div className="mt-8">
                      <a
                        href={lab.lab_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center text-sm font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
                      >
                        Visit Lab Website
                        <svg className="ml-1.5 w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
