class Orchestrator:
    """
    A controller that manages the full pipeline and exposes it via CLI:

    - Accepts commands like:

        - search <query> — runs a search and prints ranked results
        - ask <query> — runs the full response pipeline and prints structured output
        - intent <query> — prints only the intent classification
        - exit — quits
    - Handles error cases (empty query, no results found, etc.)
    - Logs each interaction to a file (session.log)
    """