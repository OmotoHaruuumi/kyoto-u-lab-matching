"use server";

export type ChunkMatch = {
  chunk_text: string;
  source_type: string;
  combined_score: number;
};

export type LabResult = {
  lab_id: number;
  name: string;
  name_en: string | null;
  department: string | null;
  faculty: string | null;
  lab_url: string | null;
  description: string | null;
  keywords: string[] | null;
  matched_chunks: ChunkMatch[];
  total_score: number;
};

export type SearchResponse = {
  query: string;
  results: LabResult[];
};

export async function searchLabs(query: string): Promise<SearchResponse | { error: string }> {
  if (!query) return { query: "", results: [] };

  try {
    const res = await fetch(`http://localhost:8000/api/v1/search?q=${encodeURIComponent(query)}&limit=10`, {
      method: "GET",
      // Set to no-store to ensure we do dynamic fetching
      cache: "no-store", 
    });

    if (!res.ok) {
      console.error("Backend error:", res.status, res.statusText);
      return { error: "Failed to fetch search results. Please ensure the backend is running." };
    }

    const data: SearchResponse = await res.json();
    return data;
  } catch (err: any) {
    console.error("Fetch error:", err);
    return { error: "Could not connect to the search API. Is the backend running?" };
  }
}
