"""Vector store setup and management for the Nomad Travel Agent."""

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings as ChromaEmbeddings
import google.generativeai as genai
import google.generativeai.types as genai_types
import os
import pandas as pd
import numpy as np
import uuid
import traceback
from typing import List, Dict, Any, Optional, Tuple
from retry import retry

from . import config
from .tools import is_retriable_google_sdk

class GeminiEmbeddingFunctionChroma(EmbeddingFunction):
    """Custom embedding function for ChromaDB using Gemini API."""
    def __init__(self, client: genai.GenerativeModel, model_name: str = config.EMBEDDING_MODEL_NAME, task_type: str = "retrieval_document"):
        self._client = client
        self._model_name = model_name
        self._task_type = task_type
        self._embed_content_with_retry = retry(predicate=is_retriable_google_sdk)(self._client.embed_content)
        print(f"GeminiEmbeddingFunctionChroma initialized for task: {self._task_type}")

    def __call__(self, input_texts: Documents) -> ChromaEmbeddings:
        """Generates embeddings for the input texts."""
        embeddings = []
        for text in input_texts:
            try:
                result = self._embed_content_with_retry(text, task_type=self._task_type)
                if result and result.embedding:
                    embeddings.append(result.embedding)
                else:
                    # Fallback to random embedding if needed
                    embeddings.append(np.random.rand(768).tolist())  # Adjust dimension as needed
            except Exception as e:
                print(f"Error generating embedding: {e}")
                embeddings.append(np.random.rand(768).tolist())  # Fallback
        return embeddings

def setup_chroma_db(df_processed: pd.DataFrame, genai_client: genai.GenerativeModel) -> Tuple[Optional[chromadb.Collection], Optional[GeminiEmbeddingFunctionChroma]]:
    """Sets up ChromaDB, indexes data if needed, and returns collection and query embedder."""
    print("\n--- Setting up ChromaDB ---")
    if df_processed is None or df_processed.empty:
        print("Error: Processed DataFrame is empty. Cannot setup ChromaDB.")
        return None, None
    if 'content_for_rag' not in df_processed.columns:
        print("Error: 'content_for_rag' column missing. Cannot setup ChromaDB.")
        return None, None
    if genai_client is None:
        print("Error: GenAI client not available for embedding. Cannot setup ChromaDB.")
        return None, None

    try:
        if not os.path.exists(config.CHROMA_CACHE_DIR):
            os.makedirs(config.CHROMA_CACHE_DIR)
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_CACHE_DIR)
        print(f"ChromaDB PersistentClient initialized at: {config.CHROMA_CACHE_DIR}")

        # Initialize embedders
        gemini_embedder_docs = GeminiEmbeddingFunctionChroma(client=genai_client, task_type="retrieval_document")
        gemini_embedder_query = GeminiEmbeddingFunctionChroma(client=genai_client, task_type="retrieval_query")

        # Get or create collection
        poi_collection = chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=gemini_embedder_docs,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{config.COLLECTION_NAME}' retrieved or created.")

        # Check if indexing is needed
        df_filtered = df_processed.dropna(subset=['content_for_rag', 'city', 'name'])
        df_filtered = df_filtered[df_filtered['content_for_rag'].str.strip() != ""]
        expected_count = len(df_filtered)
        existing_count = poi_collection.count()

        if existing_count < (expected_count * 0.98):  # Allow for some tolerance
            print(f"Indexing {expected_count} documents...")
            documents_to_index = df_filtered['content_for_rag'].tolist()
            metadatas_to_index = df_filtered[['city', 'name', 'poi_type']].astype(str).to_dict('records')
            ids_to_index = [f"poi_{uuid.uuid4()}" for _ in range(len(documents_to_index))]

            # Batch processing
            batch_size = 100
            for i in range(0, len(documents_to_index), batch_size):
                end_idx = min(i + batch_size, len(documents_to_index))
                try:
                    poi_collection.add(
                        documents=documents_to_index[i:end_idx],
                        metadatas=metadatas_to_index[i:end_idx],
                        ids=ids_to_index[i:end_idx]
                    )
                    print(f"Indexed batch {i//batch_size + 1}/{(len(documents_to_index) + batch_size - 1)//batch_size}")
                except Exception as batch_error:
                    print(f"Error indexing batch {i//batch_size + 1}: {batch_error}")
                    continue

        print("✅ ChromaDB setup complete.")
        return poi_collection, gemini_embedder_query

    except Exception as e:
        print(f"❌ Error during ChromaDB setup: {e}")
        traceback.print_exc()
        return None, None

def retrieve_rag_documents(
    city: str,
    preferences: Dict[str, Any],
    collection: Optional[chromadb.Collection],
    embedder: Optional[EmbeddingFunction],
    n_results: int = config.RAG_N_RESULTS
) -> List[str]:
    """Retrieves relevant POI documents from ChromaDB."""
    if collection is None or embedder is None:
        print("Error: ChromaDB collection or embedder not available.")
        return []

    try:
        # Construct query from city and preferences
        query_parts = [f"Find information about {city}"]
        for key, value in preferences.items():
            if value:
                query_parts.append(f"{key}: {value}")
        query = " ".join(query_parts)

        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"city": city} if city else None
        )

        # Extract and return documents
        if results and results['documents']:
            return results['documents'][0]  # First query's results
        return []

    except Exception as e:
        print(f"Error retrieving RAG documents: {e}")
        return [] 