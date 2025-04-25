import chromadb
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings as ChromaEmbeddings,
)
import google.genai as genai
import google.genai.types as genai_types
import os
import shutil
import pandas as pd
import traceback
from typing import List, Dict, Any, Optional, Tuple
from retry import retry
import google.api_core.exceptions

from . import config
from .tools import get_google_retryable_exceptions


class GeminiEmbeddingFunctionChroma(EmbeddingFunction):
    """Custom embedding function for ChromaDB using Gemini API."""

    def __init__(
        self,
        client: genai.Client,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        task_type: str = "retrieval_document",
    ):
        self._client = client
        self._model_name = model_name
        self._task_type = task_type
        try:

            self._embed_content_with_retry = retry(
                exceptions=get_google_retryable_exceptions(),
                tries=3,
                delay=1,
                backoff=2,
            )(self._client.models.embed_content)
            print("   Retry logic configured for embed_content.")
        except Exception as e:
            print(
                f"⚠️ Warning: Failed to configure retry logic for embed_content: {e}. Using direct call."
            )
            self._embed_content_with_retry = self._client.embed_content

    def __call__(self, input_texts: Documents) -> ChromaEmbeddings:
        """Embeds a list of text documents using client.embed_content."""
        if not input_texts:
            return []
        if not isinstance(input_texts, list):
            print(
                f"Embedder Input Error: input_texts is not a list (type: {type(input_texts)})."
            )

            return [
                []
                for _ in range(
                    len(input_texts) if hasattr(input_texts, "__len__") else 1
                )
            ]
        if not all(isinstance(text, str) for text in input_texts):
            print(f"Embedder Input Error: Not all items in input_texts are strings.")
            return [[] for _ in range(len(input_texts))]

        try:
            response = self._embed_content_with_retry(
                model=self._model_name,
                contents=input_texts,
                config=genai_types.EmbedContentConfig(task_type=self._task_type),
            )

            if hasattr(response, "embeddings") and isinstance(
                response.embeddings, list
            ):
                embeddings_list = []
                valid_embeddings_count = 0
                for i, emb_obj in enumerate(response.embeddings):

                    if (
                        hasattr(emb_obj, "values")
                        and isinstance(emb_obj.values, list)
                        and emb_obj.values
                    ):
                        embeddings_list.append(list(map(float, emb_obj.values)))
                        valid_embeddings_count += 1
                    else:
                        print(
                            f"DEBUG EMBED: WARNING - Item {i} in batch missing/invalid 'values'. Type: {type(emb_obj)}. Replacing with zeros."
                        )

                        embeddings_list.append([0.0] * 768)

                if valid_embeddings_count != len(input_texts):
                    print(
                        f"DEBUG EMBED: WARNING - Mismatch. Input: {len(input_texts)}, Valid Extracted: {valid_embeddings_count}."
                    )

                return embeddings_list
            else:
                print(
                    f"DEBUG EMBED: ERROR - Response object missing 'embeddings' list or invalid structure."
                )

                return [[] for _ in range(len(input_texts))]

        except google.api_core.exceptions.GoogleAPIError as api_err:
            print(
                f"DEBUG EMBED: ERROR - Google API Error during batch embedding (Task: '{self._task_type}'). Details: {api_err}"
            )

            return [[] for _ in range(len(input_texts))]
        except Exception as e:
            print(
                f"DEBUG EMBED: ERROR - Unexpected error during batch embedding (Task: '{self._task_type}'). Details: {e}"
            )
            traceback.print_exc()

            return [[] for _ in range(len(input_texts))]


def setup_chroma_db(
    df_processed: pd.DataFrame,
    genai_client: genai.Client,
) -> Tuple[Optional[chromadb.Collection], Optional[GeminiEmbeddingFunctionChroma]]:
    """Sets up ChromaDB, indexes data if needed, and returns collection and query embedder."""
    print("\n--- Setting up ChromaDB ---")

    # Validate input parameters
    if df_processed is None or df_processed.empty:
        print("Error: Processed DataFrame is empty. Cannot setup ChromaDB.")
        return None, None
    if "content_for_rag" not in df_processed.columns:
        print("Error: 'content_for_rag' column missing. Cannot setup ChromaDB.")
        return None, None
    if genai_client is None:
        print("Error: GenAI client not available for embedding. Cannot setup ChromaDB.")
        return None, None

    # Create directory if it doesn't exist
    os.makedirs(config.CHROMA_CACHE_DIR, exist_ok=True)

    try:
        # Initialize ChromaDB client with fallback to in-memory
        try:
            chroma_client = chromadb.PersistentClient(path=config.CHROMA_CACHE_DIR)
            print(
                f"ChromaDB PersistentClient initialized at: {config.CHROMA_CACHE_DIR}"
            )

            # Initialize embedders
            gemini_embedder_docs = GeminiEmbeddingFunctionChroma(
                client=genai_client, task_type="retrieval_document"
            )
            gemini_embedder_query = GeminiEmbeddingFunctionChroma(
                client=genai_client, task_type="retrieval_query"
            )

            # Check for database integrity by trying to get collection
            try:
                poi_collection = chroma_client.get_or_create_collection(
                    name=config.COLLECTION_NAME,
                    embedding_function=gemini_embedder_docs,
                    metadata={"hnsw:space": "cosine"},
                )

                # Test if the collection is accessible with a simple query
                _ = poi_collection.query(query_embeddings=[[0.0] * 768], n_results=1)
                print(f"✅ Collection '{config.COLLECTION_NAME}' verified.")

            except Exception as coll_err:
                print(f"❌ Database integrity check failed: {coll_err}")
                print("🔄 Recreating database due to corruption...")
                shutil.rmtree(config.CHROMA_CACHE_DIR)
                os.makedirs(config.CHROMA_CACHE_DIR)

                # Reinitialize client and collection after recreation
                chroma_client = chromadb.PersistentClient(path=config.CHROMA_CACHE_DIR)
                poi_collection = chroma_client.get_or_create_collection(
                    name=config.COLLECTION_NAME,
                    embedding_function=gemini_embedder_docs,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"Collection '{config.COLLECTION_NAME}' recreated successfully.")

        except Exception as client_err:
            print(f"Error initializing PersistentClient: {client_err}")
            print("Falling back to in-memory client...")
            chroma_client = chromadb.Client()

            # Initialize embedders (again in case of exception)
            gemini_embedder_docs = GeminiEmbeddingFunctionChroma(
                client=genai_client, task_type="retrieval_document"
            )
            gemini_embedder_query = GeminiEmbeddingFunctionChroma(
                client=genai_client, task_type="retrieval_query"
            )

            # Create collection with in-memory client
            poi_collection = chroma_client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                embedding_function=gemini_embedder_docs,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Collection '{config.COLLECTION_NAME}' created in memory.")

        # Prepare data for indexing
        df_filtered = df_processed.dropna(subset=["content_for_rag", "city", "name"])
        df_filtered = df_filtered[df_filtered["content_for_rag"].str.strip() != ""]
        expected_count = len(df_filtered)
        existing_count = poi_collection.count()
        print(
            f"Expected documents: {expected_count}, Found in collection: {existing_count}"
        )

        # Determine if we need to populate more data
        count_threshold = 0.98
        if (
            existing_count >= int(expected_count * count_threshold)
            and existing_count > 0
        ):
            print(
                f"✅ Collection has sufficient data ({existing_count}). Loading from cache."
            )
        elif expected_count == 0:
            print(
                "⚠️ Warning: No processable documents found in DataFrame. Collection will be empty."
            )
        else:
            print(
                f"ℹ️ Collection has {existing_count} documents, expecting {expected_count}."
            )

            # Filter to only process documents not already in the database
            if existing_count > 0:
                print(f"Identifying new documents to add...")

                # Get existing documents to detect duplicates
                max_sample = min(10000, existing_count)
                print(
                    f"Sampling up to {max_sample} existing documents to identify duplicates..."
                )

                # Query existing documents in batches
                existing_keys = set()
                batch_size_query = 1000

                for i in range(0, max_sample, batch_size_query):
                    sample_size = min(batch_size_query, max_sample - i)
                    print(
                        f"  Retrieving batch {i//batch_size_query + 1} with {sample_size} documents..."
                    )

                    # Generate random embeddings for querying (just to get samples)
                    dummy_embeddings = [[0.1] * 768]

                    # Get sample of documents from collection
                    sample_results = poi_collection.query(
                        query_embeddings=dummy_embeddings,
                        n_results=sample_size,
                        include=["metadatas"],
                        offset=i,
                    )

                    # Extract metadata
                    if (
                        sample_results
                        and sample_results.get("metadatas")
                        and sample_results["metadatas"][0]
                    ):
                        batch_keys = [
                            (metadata.get("city", ""), metadata.get("name", ""))
                            for metadata in sample_results["metadatas"][0]
                            if metadata
                        ]

                        # Add to set of existing keys
                        existing_keys.update(
                            [key for key in batch_keys if key[0] and key[1]]
                        )

                print(
                    f"  Found {len(existing_keys)} unique city/name combinations in database"
                )

                # Create a composite key for faster lookups
                df_filtered["composite_key"] = df_filtered.apply(
                    lambda row: (str(row["city"]), str(row["name"])), axis=1
                )

                # Filter dataframe to only include new entries
                df_filtered["is_new"] = ~df_filtered["composite_key"].isin(
                    existing_keys
                )
                new_docs_count = df_filtered["is_new"].sum()

                if new_docs_count > 0:
                    print(
                        f"Found {new_docs_count} new documents to add (out of {len(df_filtered)})"
                    )
                    df_filtered = df_filtered[df_filtered["is_new"]].drop(
                        ["composite_key", "is_new"], axis=1
                    )
                else:
                    print("No new documents identified to add")
                    df_filtered = df_filtered.head(0)  # Empty dataframe

            if len(df_filtered) > 0:
                print(
                    "Proceeding with embedding and indexing new documents (this may take time)..."
                )

                # Prepare documents for indexing
                documents_to_index = df_filtered["content_for_rag"].tolist()
                metadatas_to_index = (
                    df_filtered[["city", "name", "poi_type"]]
                    .astype(str)
                    .to_dict("records")
                )

                # Generate stable IDs based on content to avoid duplicates
                import hashlib

                def generate_stable_id(city, name, content):
                    """Generate stable ID from content to avoid duplicates"""
                    hash_input = f"{city}:{name}:{content[:100]}"
                    return f"poi_{hashlib.md5(hash_input.encode()).hexdigest()}"

                ids_to_index = [
                    generate_stable_id(
                        df_filtered.iloc[i]["city"],
                        df_filtered.iloc[i]["name"],
                        df_filtered.iloc[i]["content_for_rag"],
                    )
                    for i in range(len(documents_to_index))
                ]

                # Configure batch processing
                batch_size = 100
                added_count = 0
                total_batches = (len(documents_to_index) + batch_size - 1) // batch_size
                print(
                    f"Adding {len(documents_to_index)} documents in {total_batches} batches of size {batch_size}..."
                )

                # Process in batches with adaptive error handling
                for i in range(0, len(documents_to_index), batch_size):
                    batch_docs = documents_to_index[i : i + batch_size]
                    batch_metadatas = metadatas_to_index[i : i + batch_size]
                    batch_ids = ids_to_index[i : i + batch_size]
                    current_batch_num = (i // batch_size) + 1

                    print(
                        f"  Adding batch {current_batch_num}/{total_batches} ({len(batch_docs)} documents)..."
                    )

                    try:
                        # Calculate document stats for debugging
                        max_len = (
                            max(len(doc) for doc in batch_docs) if batch_docs else 0
                        )
                        avg_len = (
                            sum(len(doc) for doc in batch_docs) / len(batch_docs)
                            if batch_docs
                            else 0
                        )
                        print(
                            f"  Batch stats - Max doc length: {max_len}, Avg length: {avg_len:.1f}"
                        )

                        # Add documents
                        poi_collection.add(
                            documents=batch_docs,
                            metadatas=batch_metadatas,
                            ids=batch_ids,
                        )
                        added_count += len(batch_docs)

                    except Exception as batch_error:
                        print(
                            f"  ❌ Error adding Batch {current_batch_num}: {batch_error}"
                        )

                        # Try with smaller batch size if possible
                        half_batch = len(batch_docs) // 2
                        if half_batch > 0:
                            print(
                                f"  Attempting to add in smaller batches ({half_batch} documents)..."
                            )
                            try:
                                # Add first half
                                poi_collection.add(
                                    documents=batch_docs[:half_batch],
                                    metadatas=batch_metadatas[:half_batch],
                                    ids=batch_ids[:half_batch],
                                )
                                # Add second half
                                poi_collection.add(
                                    documents=batch_docs[half_batch:],
                                    metadatas=batch_metadatas[half_batch:],
                                    ids=batch_ids[half_batch:],
                                )
                                added_count += len(batch_docs)
                                print(
                                    f"  ✅ Successfully added with reduced batch size"
                                )
                            except Exception as e:
                                print(f"  ❌ Failed even with reduced batch: {e}")
                                print(f"     Skipping this batch due to error.")
                        else:
                            print(f"     Skipping this batch due to error.")

                        continue

                # Report indexing results
                print(f"\n✅ Indexing complete.")
                print(f"   Attempted to add {len(documents_to_index)} documents.")
                print(f"   Successfully added ~{added_count} documents in this run.")
                final_count = poi_collection.count()
                print(
                    f"   Collection '{config.COLLECTION_NAME}' now contains {final_count} documents."
                )

                if final_count < expected_count * count_threshold:
                    print(
                        f"   ⚠️ Warning: Final count ({final_count}) is lower than expected ({expected_count})."
                    )
                    print(
                        f"      This could be due to errors during batch processing or duplicate documents."
                    )
            else:
                print("No new documents to add.")

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
    n_results: int = config.RAG_N_RESULTS,
) -> List[str]:
    """Retrieves relevant POI documents from ChromaDB."""
    if collection is None or embedder is None:
        print("Error: ChromaDB collection or embedder not available for RAG.")
        return []
    if not city or not isinstance(city, str):
        print(f"RAG Error: Invalid city provided: {city}")
        return []
    if not preferences or not isinstance(preferences, dict):
        print(f"RAG Error: Invalid preferences provided: {preferences}")
        return []

    pref_list = [f"{k}: {v}" for k, v in preferences.items() if v]
    pref_string = ", ".join(pref_list)
    query_text = f"Points of interest in {city} suitable for someone interested in [{pref_string}]"

    try:

        query_embedding_list = embedder([query_text])

        if (
            not query_embedding_list
            or not isinstance(query_embedding_list, list)
            or len(query_embedding_list) == 0
            # or not query_embedding_list[0]
        ):
            print(
                f"RAG Error: Embedder failed to return a valid embedding for the query."
            )
            return []

        query_vector = query_embedding_list[0]

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where={"city": city},
        )

        if (
            results
            and isinstance(results.get("documents"), list)
            and results["documents"]
        ):

            return results["documents"][0]
        else:

            return []

    except Exception as e:
        print(f"❌ Error retrieving RAG documents: {e}")
        traceback.print_exc()
        return []
