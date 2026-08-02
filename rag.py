import re

import numpy as np


class RAG:
    """
    Implement a Retrieval-Augmented Generation system from scratch:

    - Parse and chunk the policy document into overlapping text chunks
    - Build a TF-IDF vocabulary matrix using numpy — no sklearn
    - Implement cosine similarity search using numpy to retrieve the top-K most relevant chunks for a query
    - Store chunks and their TF-IDF vectors in memory (your "vector store")
    """
    def __init__(self, filepath:str = "policy.txt"):
        self.filepath = filepath
        self.chunks: list[str] = []
        self.vocab_list: list[str] = []

        # tf-idf assets
        self.idf_vector: np.ndarray = np.array([])
        self.tf_idf_matrix: np.ndarray = np.array([])

        # bm25 assets
        self.bm25_idf_vector: np.ndarray = np.array([])
        self.raw_tf_matrix: np.ndarray = np.array([])
        self.chunk_lengths: np.ndarray = np.array([])
        self.avg_chunk_length: float = 0.0

        self.fit()

    def create_chunks(self, chunk_size: int = 50, overlap: int = 10) -> list[str]:
        """
        Create chunks list from the provided document.
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size should be less than overlap.")
        
        with open(self.filepath, 'r') as f:
            data = f.read()
        
        words = [word for word in re.split(r'\s+', data) if word]

        chunks: list[str] = []
        
        for i in range(0, len(words), chunk_size-overlap):
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

        self.chunks = chunks
        return chunks
    
    def create_vocab(self) -> list[str]:
        """
        Create vocabulary across all chunks. In each chunks the number of unique elements may vary. Applying cosine similarity later this vocab will be used. for every chunk tf vector will be same length with absent word having 0.0 as value 
        """
        vocab_set: set[str] = set()
        for chunk in self.chunks:
            words = [w.lower() for w in re.split(r'\s+', chunk) if w]
            vocab_set.update(words)

        self.vocab_list = sorted(vocab_set)
        return self.vocab_list

    def build_matrices(self):
        """
        Builds prerequisites for both TF-IDF and BM25 simultaneously 
        to avoid processing the chunks multiple times.
        bm25 = idf*((raw_tf*(term_freq_saturation_param+1))/raw_tf + term_freq_saturation_param*(1-length normalization param+b*length of current chunk/avg chunk length across entire dataset)
        """
        tokenized_chunks: list[list[str]] = []
        chunk_lengths_list: list[int] = []

        for chunk in self.chunks:
            words = [w.lower() for w in re.split(r'\s+', chunk) if w]
            tokenized_chunks.append(words)
            chunk_lengths_list.append(len(words))

        self.chunk_lengths = np.array(chunk_lengths_list, dtype=np.float64)
        self.avg_chunk_length = float(np.mean(self.chunk_lengths)) if len(self.chunk_lengths) > 0 else 1.0

        # build raw freq matrix (rows = chunks, cols = vocab words)
        raw_tf_list: list[list[float]] = []
        for words in tokenized_chunks:
            local_counts: dict[str, int] = {}
            for word in words:
                local_counts[word] = local_counts.get(word,0)+1
            current_chunk_counts = [float(local_counts.get(vocab_word, 0)) for vocab_word in self.vocab_list]
            raw_tf_list.append(current_chunk_counts)

        self.raw_tf_matrix = np.array(self.raw_tf_matrix, dtype=np.float64)

        # compute tf-idf
        total_chunks = len(tokenized_chunks)
        chunks_containing_term = np.sum(self.raw_tf_matrix>0, axis=0)
        # compute tf-idf log calculations
        self.idf_vector = np.log(total_chunks / (1 + chunks_containing_term))
        # normalized tf for cosine similarity
        normalized_tf = self.raw_tf_matrix / np.maximum(self.chunk_lengths[:,np.newaxis], 1.0)
        self.tf_idf_matrix = normalized_tf * self.idf_vector
        # compute bm25 idf variant
        # Lucene/BM25 traditional IDF uses a smoothed variant that can go negative if a term is in >50% of docs.
        # We wrap in np.maximum(..., 1e-5) to prevent negative weights for incredibly common terms.
        bm25_idf = np.log((total_chunks - chunks_containing_term + 0.5) / (chunks_containing_term + 0.5) + 1.0)
        self.bm25_idf_vector = np.maximum(bm25_idf, 1e-5)
 
    def create_tf_idf_matrix(self):
        """
        Create tf(term-frequency) vector - for each chunk, count how often each word appears ÷ total words in chunk
        Create idf(inverse document frequency) vector - log(total_chunks / chunks_containing_term) — penalises common words
        """
        tokenized_chunks: list[list[str]] = []
        for chunk in self.chunks:
            words = [w.lower() for w in re.split(r'\s+', chunk) if w]
            tokenized_chunks.append(words)

        tf_matrix_list: list[list[float]] = []
        for words in tokenized_chunks:
            total_words_in_chunk = len(words)

            if total_words_in_chunk == 0:
                current_chunk_tf = [0.0]*len(self.vocab_list)
                tf_matrix_list.append(current_chunk_tf)
                continue

            local_counts: dict[str,int] = {}
            for word in words:
                local_counts[word] = local_counts.get(word, 0) + 1

            current_chunk_tf: list[float] = []
            for vocab_word in self.vocab_list:
                word_count = local_counts.get(vocab_word, 0)
                current_chunk_tf.append(word_count/total_words_in_chunk)

            tf_matrix_list.append(current_chunk_tf)
            
        tf_matrix = np.array(tf_matrix_list, dtype=np.float64)

        total_chunks = len(tokenized_chunks)
        chunks_containing_term = np.sum(tf_matrix>0, axis=0)
        self.idf_vector = np.log(total_chunks / (1 + chunks_containing_term))

        self.tf_idf_matrix = tf_matrix * self.idf_vector
        return self.tf_idf_matrix
     
    def query_vectorization(self, query: str):
        """
        Transforms a runtime query string into an aligned 1D TF-IDF vector.
        """
        # clean and tokenize single query string
        query_words = [word.lower() for word in re.split(r'\s+', query) if word]
        total_words = len(query_words)
        
        if total_words == 0:
            return np.zeros(len(self.vocab_list), dtype=np.float64)
        
        query_counts: dict[str, int] = {}
        for word in query_words:
            query_counts[word] = query_counts.get(word, 0) + 1

        query_tf: list[float] = []
        for vocab_word in self.vocab_list:
            word_count = query_counts.get(vocab_word, 0)
            query_tf.append(word_count / total_words)

        query_tf_vector = np.array(query_tf, dtype=np.float64)

        query_tf_idf_vector = query_tf_vector * self.idf_vector

        return query_tf_idf_vector
    
    def search(self, query: str, top_k: int = 3):
        """
        Computes cosine similarities across the entire database matrix at once.
        Returns sorted tracking lists containing tuples of (chunk_index, score).
        """
        query_vec = self.query_vectorization(query)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return [(i, 0.0) for i in range(min(top_k, len(self.chunks)))]
        
        matrix_norms = np.linalg.norm(self.tf_idf_matrix, axis=1)
        # Handle zero-norm fallback for unpopulated document entries safely
        matrix_norms[matrix_norms == 0] = 1.0

        # Matrix-vector dot product computes all intersections simultaneously
        dot_products = np.dot(self.tf_idf_matrix, query_vec)
        similarities = dot_products / (matrix_norms * query_norm)

        # Extract the highest alignment indexes sorted in descending order
        ranked_indices = np.argsort(similarities)[::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in ranked_indices[:top_k]]
    
    def search_tf_idf(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        query_words = [word.lower() for word in re.split(r'\s+', query) if word]
        total_words = len(query_words)

        if total_words == 0:
            return [(i, 0.0) for i in range(min(top_k, len(self.chunks)))]
        
        query_counts: dict[str, int] = {}
        for word in query_words:
            query_counts[word] = query_counts.get(word, 0) + 1

        query_tf = [query_counts.get(vocab_word,0) / total_words for vocab_word in self.vocab_list]
        query_vec = np.array(query_tf, dtype=np.float64) * self.idf_vector

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return [(i,0.0) for i in range(min(top_k, len(self.chunks)))]
        
        matrix_norms = np.linalg.norm(self.tf_idf_matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1.0

        dot_products = np.dot(self.tf_idf_matrix, query_vec)
        similarities = dot_products / (matrix_norms * query_norm)
        ranked_indices = np.argsort(similarities)[::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in ranked_indices[:top_k]]

    def search_bm25(self, query: str, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
        """
        Computes scores using the BM25 probabilistic relevance algorithm.
        """
        query_words = [word.lower() for word in re.split(r'\s+', query) if word]
        if not query_words or len(self.chunks) == 0:
            return [(i, 0.0) for i in range(min(top_k, len(self.chunks)))]

        # Map query tokens directly to vocabulary array indices
        vocab_index_map = {word: i for i, word in enumerate(self.vocab_list)}
        query_indices = [vocab_index_map[w] for w in query_words if w in vocab_index_map]

        if not query_indices:
            return [(i, 0.0) for i in range(min(top_k, len(self.chunks)))]

        # Isolate the columns of our raw_tf matrix matching the query terms
        # Matrix shape: (num_chunks, num_query_terms_found)
        tf_q = self.raw_tf_matrix[:, query_indices]
        idf_q = self.bm25_idf_vector[query_indices]

        # Calculate the length normalization denominator component: 1 - b + b * (dl / avgdl)
        # Vector shape: (num_chunks, 1)
        len_norm = (1.0 - b) + b * (self.chunk_lengths / self.avg_chunk_length)
        len_norm = len_norm[:, np.newaxis] 

        # Compute the fractional term scaling for all documents simultaneously
        # Formula: tf * (k1 + 1) / (tf + k1 * len_norm)
        score_matrix = (tf_q * (k1 + 1.0)) / (tf_q + k1 * len_norm)

        # Multiply by term IDFs and sum across terms to get final scores per chunk
        bm25_scores = np.sum(score_matrix * idf_q, axis=1)

        ranked_indices = np.argsort(bm25_scores)[::-1]
        return [(int(idx), float(bm25_scores[idx])) for idx in ranked_indices[:top_k]]

    def fit(self, chunk_size: int = 50, overlap: int = 10):
        self.create_chunks(chunk_size, overlap)
        self.create_vocab()
        self.build_matrices()
        self.create_tf_idf_matrix()
        