import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import openai
from flask import current_app

# --- Document Partitioning Imports ---
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.csv import partition_csv
from unstructured.partition.doc import partition_doc
from unstructured.partition.docx import partition_docx
from unstructured.partition.ppt import partition_ppt
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.html import partition_html
# Add others as needed

from app.utils import get_file_extension # Import utility

# --- Document Partitioning Service ---

PARTITION_FUNCTIONS = {
    ".pdf": partition_pdf,
    ".docx": partition_docx,
    ".doc": partition_doc, # May require extra dependencies (libreoffice, pandoc)
    ".pptx": partition_pptx,
    ".ppt": partition_ppt,  # May require extra dependencies
    ".csv": partition_csv,
    ".html": partition_html,
    ".htm": partition_html,
}

def partition_document_service(file_path, original_filename):
    """
    Partitions a document based on its file extension using unstructured.

    Args:
        file_path (str): The path to the temporary document file.
        original_filename (str): The original name of the uploaded file.

    Returns:
        list: A list of unstructured Elements.

    Raises:
        ValueError: If the file extension is unsupported.
        ImportError: If a required dependency for the file type is missing.
        RuntimeError: For general partitioning errors.
    """
    file_extension = get_file_extension(original_filename) # Use original filename for type
    partition_func = PARTITION_FUNCTIONS.get(file_extension)

    if not partition_func:
        current_app.logger.error(f"Unsupported file type: {file_extension} for file {original_filename}")
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        current_app.logger.info(f"Partitioning document: {original_filename} (path: {file_path}) using {partition_func.__name__}")
        # Consider adding strategy="hi_res" for PDFs if needed, requires Detectron2/poppler
        elements = partition_func(filename=file_path)
        current_app.logger.info(f"Successfully partitioned {original_filename} into {len(elements)} elements.")
        return elements
    except ImportError as ie:
         current_app.logger.error(f"ImportError during partitioning {original_filename}. Missing dependency for {file_extension}? Error: {ie}", exc_info=True)
         raise ImportError(f"A required library for processing '{file_extension}' files might be missing (e.g., for .doc or hi-res PDF). Please check unstructured documentation. Original error: {ie}")
    except Exception as e:
        current_app.logger.error(f"Failed to partition document {original_filename}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to partition document: {e}")


# --- Vector Store Service ---

_chroma_client = None
_embedding_function = None

def _get_embedding_function():
    """Initializes and returns the embedding function based on config (cached)."""
    global _embedding_function
    if _embedding_function is None:
        conf = current_app.config
        ef_type = conf.get('EMBEDDING_API_TYPE', '').lower()
        api_key = conf.get('EMBEDDING_API_KEY')
        api_base = conf.get('EMBEDDING_API_BASE')
        api_version = conf.get('EMBEDDING_API_VERSION')
        model_name = conf.get('EMBEDDING_DEPLOYMENT')

        if api_key and api_base and api_version and model_name and (ef_type == 'openai' or ef_type == 'azure'):
            current_app.logger.info(f"Using OpenAI Embedding Function (Type: {ef_type}, Model: {model_name})")
            _embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=api_base,
                api_type=ef_type,
                api_version=api_version,
                model_name=model_name
            )
        else:
            current_app.logger.warning(f"OpenAI/Azure Embedding Function config incomplete or missing. Falling back to default SentenceTransformer. Ensure 'sentence-transformers' is installed.")
            try:
                 # Ensure sentence-transformers is installed: pip install sentence-transformers
                 _embedding_function = embedding_functions.DefaultEmbeddingFunction()
            except ImportError:
                 current_app.logger.error("DefaultEmbeddingFunction requires 'sentence-transformers'. Please install it.", exc_info=True)
                 raise ImportError("Default embedding function requires 'sentence-transformers'. Install it or configure OpenAI/Azure embeddings.")
    return _embedding_function

def _get_chroma_client():
    """Initializes and returns a persistent ChromaDB client (cached)."""
    global _chroma_client
    if _chroma_client is None:
        chromadb_path = current_app.config['CHROMADB_FOLDER']
        current_app.logger.info(f"Initializing ChromaDB client. Path: {chromadb_path}")
        _chroma_client = chromadb.PersistentClient(path=chromadb_path)
    return _chroma_client


def add_document_chunks_service(collection_name, documents, ids, metadatas=None):
    """
    Adds document chunks (with embeddings) to a ChromaDB collection.

    Args:
        collection_name (str): Name of the collection.
        documents (list[str]): List of document text chunks.
        ids (list[str]): List of unique IDs for each chunk.
        metadatas (list[dict], optional): List of metadata dictionaries for each chunk.

    Returns:
        bool: True if successful.

    Raises:
        ValueError: If input lists are mismatched or invalid.
        RuntimeError: If embedding or upsertion fails.
    """
    if not documents or not ids or len(documents) != len(ids):
        raise ValueError("Mismatch between documents and ids count or empty lists provided.")
    if metadatas and len(documents) != len(metadatas):
         raise ValueError("Mismatch between documents and metadatas count.")

    client = _get_chroma_client()
    embedding_func = _get_embedding_function()

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_func # Associate EF with collection
        )

        # Upsert - ChromaDB handles embedding calculation via the associated function
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        current_app.logger.info(f"Successfully upserted {len(ids)} chunks into collection '{collection_name}'.")
        return True
    except Exception as e:
        current_app.logger.error(f"Error adding chunks to collection '{collection_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to add chunks to vector store: {e}")


def query_vector_store_service(collection_name, query_text, n_results=5, where_filter=None):
    """
    Queries a ChromaDB collection for relevant documents.

    Args:
        collection_name (str): Name of the collection to query.
        query_text (str): The query text.
        n_results (int): Number of results to return.
        where_filter (dict, optional): A filter dictionary for metadata.

    Returns:
        dict: The query results from ChromaDB.

    Raises:
        ValueError: If the collection is not found.
        RuntimeError: If the query fails for other reasons.
    """
    client = _get_chroma_client()
    embedding_func = _get_embedding_function() # EF needed for query embedding consistency

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func # Ensures consistency
            )

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count()), # Avoid asking for more results than exist
            where=where_filter if where_filter else None, # Pass None if empty
            include=['documents', 'metadatas', 'distances']
        )
        num_found = len(results.get('ids', [[]])[0])
        current_app.logger.info(f"Query to '{collection_name}' for '{query_text[:50]}...' found {num_found} results (asked for {n_results}).")
        return results
    except chromadb.errors.CollectionNotFoundError: # More specific exception handling is good
         current_app.logger.warning(f"Collection '{collection_name}' not found for querying.")
         raise ValueError(f"Collection '{collection_name}' does not exist.")
    except ValueError as ve: # Catch potential embedding errors during query
         current_app.logger.error(f"Value error during query (likely embedding issue): {ve}", exc_info=True)
         raise RuntimeError(f"Failed to query vector store due to value error: {ve}")
    except Exception as e:
        current_app.logger.error(f"Error querying collection '{collection_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to query vector store: {e}")

# --- LLM Service ---

def _configure_openai_api():
    """Configures the OpenAI API client based on Flask app config. Should be called before API usage."""
    conf = current_app.config
    api_key = conf.get('GPT_API_KEY')
    api_base = conf.get('GPT_API_BASE')
    api_type = conf.get('GPT_API_TYPE', '').lower()
    api_version = conf.get('GPT_API_VERSION')

    if not all([api_key, api_base, api_type, api_version]):
        raise ValueError("Missing required OpenAI configuration (Key, Base, Type, Version).")

    openai.api_key = api_key
    openai.api_base = api_base
    openai.api_type = api_type
    openai.api_version = api_version
    current_app.logger.debug(f"OpenAI API configured: Type={api_type}, Base={api_base}, Version={api_version}")

def generate_completion_service(messages, max_tokens=1500, temperature=0.7):
    """
    Generates a chat completion using the configured OpenAI model.

    Args:
        messages (list[dict]): Chat messages [{"role": "user", "content": "..."}].
        max_tokens (int): Max response tokens.
        temperature (float): Sampling temperature.

    Returns:
        str: The assistant's reply content.

    Raises:
        ValueError: If configuration is missing.
        ConnectionError: If the API call fails due to connection or auth issues.
        RuntimeError: For other LLM generation errors.
    """
    try:
        _configure_openai_api() # Ensure API is configured
    except ValueError as ve:
        current_app.logger.error(f"LLM Configuration Error: {ve}")
        raise ve # Re-raise configuration error

    model_name = current_app.config.get('GPT_DEPLOYMENT')
    if not model_name:
         raise ValueError("Missing required OpenAI configuration: GPT_DEPLOYMENT (Model Name).")

    try:
        current_app.logger.info(f"Sending request to LLM (Deployment: {model_name}) with {len(messages)} messages.")
        # Use 'engine' for Azure, 'model' for direct OpenAI (this adapts based on api_type)
        create_params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if openai.api_type == 'azure':
            create_params["engine"] = model_name
        else:
             create_params["model"] = model_name


        response = openai.ChatCompletion.create(**create_params)

        # Basic check for valid response structure
        if not response or not response.choices or not response.choices[0].message:
            current_app.logger.error(f"Invalid response structure received from LLM: {response}")
            raise RuntimeError("Received an unexpected or empty response from the LLM.")

        content = response.choices[0].message.get("content", "")
        current_app.logger.info("Received response from LLM.")
        return content.strip()

    except openai.error.AuthenticationError as e:
         current_app.logger.error(f"OpenAI Authentication Error: {e}", exc_info=True)
         raise ConnectionError(f"LLM authentication failed. Check API key/credentials: {e}")
    except openai.error.APIConnectionError as e:
         current_app.logger.error(f"OpenAI API Connection Error: {e}", exc_info=True)
         raise ConnectionError(f"Failed to connect to LLM service endpoint: {e}")
    except openai.error.InvalidRequestError as e:
         current_app.logger.error(f"OpenAI Invalid Request Error: {e}", exc_info=True)
         # Log messages for debugging if it's a request content issue
         current_app.logger.debug(f"LLM Request Messages causing error: {messages}")
         raise ValueError(f"Invalid request sent to LLM: {e}")
    except openai.error.RateLimitError as e:
        current_app.logger.error(f"OpenAI Rate Limit Error: {e}", exc_info=True)
        raise ConnectionError(f"LLM rate limit exceeded: {e}")
    except openai.error.OpenAIError as e: # Catch other OpenAI specific errors
        current_app.logger.error(f"OpenAI API Error: {e}", exc_info=True)
        raise ConnectionError(f"An error occurred with the LLM service: {e}")
    except Exception as e:
        current_app.logger.error(f"Unexpected error during LLM completion: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate completion due to an unexpected error: {e}")