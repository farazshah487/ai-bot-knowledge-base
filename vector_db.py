import faiss
import numpy as np

class FAISSVectorDB:
    def __init__(self, dimension):
        """Initialize FAISS vector index."""
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add_embeddings(self, embeddings, chunks):
        """Add embeddings and corresponding text chunks."""
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunks.extend(chunks)

    def query(self, query_embedding, top_k=3):
        """Retrieve top-k relevant chunks based on query embedding."""
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.chunks[i] for i in I[0]]
