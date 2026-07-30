from search_agent import SearchAgent


class ResponseAgent:
    """
    A class that calls the Search Agent and composes a final structured response:
    - generate_response(query: str) -> dict — returns a dict with:

    - query — original query
    - intent — from get_intent()
    - context — relevant policy excerpts
    - answer — a rule-based or template-based natural language answer derived from the context
    - confidence — a float score (0-1) based on cosine similarity of top result
    """

    def __init__(self):
        self.search_agent = SearchAgent()

    def generate_response(self, query: str):
        return self.search_agent.search(query)