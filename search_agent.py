from rag import RAG


class SearchAgent:
    """
    A class that wraps the RAG engine and exposes structured search capabilities:

    - search(query: str) -> list[dict] — returns top matching chunks with relevance scores
    - get_intent(query: str) -> str — classifies the query intent as one of: COMPLIANT, RISKY, RESTRICTED, NEUTRAL — based on keywords and context retrieved from the document
    - get_context(query: str) -> str — returns a condensed context string from top chunks
    """

    def __init__(self):
        self.rag = RAG()

    def search(self, query: str, top_k: int = 2) -> list[dict[str, str|float]]:
        """returns top matching chunks with relevance scores"""
        top_k_result = self.rag.search(query, top_k)
        chunks_with_score: list[dict[str, str|float]] = [{"chunk": self.rag.chunks[idx], "score": score} for (idx,score) in top_k_result]
        return chunks_with_score

    def get_intent(self, query: str) -> str:
        """classifies the query intent as one of: COMPLIANT, RISKY, RESTRICTED, NEUTRAL — based on keywords and context retrieved from the document"""
        return 'tbd'

    def get_context(self, query: str) -> str:
        """returns a condensed context string from top chunks"""
        return 'tbd'