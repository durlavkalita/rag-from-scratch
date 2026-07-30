from response_agent import ResponseAgent


def main():
    print("Hello from rag-from-scratch!")
    agent = ResponseAgent()

    query = "GENERATIVE AI CODE QUALITY PROTOCOLS"

    print(agent.generate_response(query))
    
    # print(f"=== Comparing Algorithms for: '{query}' ===\n")
    
    # print("--- TF-IDF (Cosine Similarity) Results ---")
    # for rank, (idx, score) in enumerate(rag.search_tfidf(query, top_k=2), 1):
    #     print(f"#{rank} | Chunk {idx} | Score: {score:.4f} | Excerpt: \"{rag.chunks[idx][:70]}...\"")
        
    # print("\n--- BM25 Scores Results ---")
    # for rank, (idx, score) in enumerate(rag.search_bm25(query, top_k=2), 1):
    #     print(f"#{rank} | Chunk {idx} | Score: {score:.4f} | Excerpt: \"{rag.chunks[idx][:70]}...\"")


if __name__ == '__main__':
    main()