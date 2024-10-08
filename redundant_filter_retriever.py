from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma
    
    def get_relevant_documents(self, query):
        # Calculate embedding for the query string
        emb = self.embeddings.embed_query(query)
        
        # Feed the embedding into the max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(embedding=emb, lambda_mult=0.8)
    
    def aget_relevant_documents(self):
        return []