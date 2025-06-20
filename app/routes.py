import os
from flask import Blueprint, request, jsonify, current_app, make_response
from werkzeug.exceptions import BadRequest, InternalServerError, NotFound, UnsupportedMediaType
from pydantic import ValidationError

from app import services  # Import the consolidated services module
from app import utils     # Import the utils module
from app.models import AskRequest, AskResponse, UploadResponse, ErrorResponse # Import Pydantic models

# Create a Blueprint
qa_bp = Blueprint('qa_bp', __name__)

# --- Helper Function for Consistent Error Responses ---
def create_error_response(message: str, status_code: int):
    current_app.logger.error(f"API Error Response ({status_code}): {message}")
    error_resp = ErrorResponse(error=message)
    # Use .dict() in Pydantic v1, .model_dump() in Pydantic v2
    # Use .model_dump() if using Pydantic v2
    return make_response(jsonify(error_resp.dict()), status_code)

# --- Error Handlers for the Blueprint ---
@qa_bp.app_errorhandler(ValidationError)
def handle_validation_error(error: ValidationError):
    # Nicely format Pydantic validation errors
    error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in error.errors()]
    return create_error_response(f"Validation Error: {'; '.join(error_messages)}", 400)

@qa_bp.app_errorhandler(FileNotFoundError)
def handle_file_not_found(error):
    return create_error_response(str(error), 404)

@qa_bp.app_errorhandler(ValueError)
def handle_value_error(error):
     # Could be collection not found, unsupported file type, invalid request data etc.
     # Check common cases
     if "Collection" in str(error) and "does not exist" in str(error):
         return create_error_response(str(error), 404) # Not Found for missing collection
     if "Unsupported file type" in str(error):
         return create_error_response(str(error), 415) # Unsupported Media Type
     return create_error_response(f"Invalid input: {str(error)}", 400) # Bad Request otherwise

@qa_bp.app_errorhandler(ImportError)
def handle_import_error(error):
    # Usually related to missing dependencies for file parsing
     return create_error_response(f"Dependency Error: {str(error)}", 501) # Not Implemented

@qa_bp.app_errorhandler(ConnectionError)
def handle_connection_error(error):
    # Errors connecting to external services (LLM, maybe embeddings)
    return create_error_response(f"Service Connection Error: {str(error)}", 503) # Service Unavailable

@qa_bp.app_errorhandler(RuntimeError)
def handle_runtime_error(error):
    # General processing errors (partitioning, embedding, LLM generation)
    return create_error_response(f"Processing Error: {str(error)}", 500) # Internal Server Error

@qa_bp.app_errorhandler(Exception)
def handle_generic_exception(error):
    # Catch-all for unexpected errors
    current_app.logger.exception("An unexpected error occurred.") # Log full traceback
    return create_error_response("An unexpected internal error occurred.", 500)


# --- API Routes ---

@qa_bp.route('/upload', methods=['POST'])
def upload_document():
    """
    Handles document upload, partitioning, embedding, and storage.
    Expects multipart/form-data with 'file' field.
    Optional 'collection_name' form field.
    """
    temp_file_path = None
    try:
        if 'file' not in request.files:
            raise BadRequest("No 'file' part in the request.")

        file = request.files['file']
        if file.filename == '':
             raise BadRequest("No selected file.")

        # --- 1. Save File Temporarily ---
        temp_file_path, original_filename = utils.save_uploaded_file(file)

        # --- 2. Determine Collection Name ---
        # Prefer user-provided name, sanitize it, otherwise derive from filename
        user_provided_name = request.form.get('collection_name', '').strip()
        if user_provided_name:
            collection_name = user_provided_name.lower().replace(" ", "_")
            current_app.logger.info(f"Using user-provided collection name: {collection_name}")
        else:
            filename_base = os.path.splitext(original_filename)[0]
            collection_name = filename_base.lower().replace(" ", "_")
            # Add a prefix for clarity, prevent potential conflicts with other data types
            collection_name = f"doc_{collection_name}"
            current_app.logger.info(f"Derived collection name: {collection_name}")

        # --- 3. Partition Document ---
        elements = services.partition_document_service(temp_file_path, original_filename)
        if not elements:
             # Should ideally be caught by partition_document_service, but double-check
             raise RuntimeError("Document partitioning yielded no results.")

        # Convert elements to strings for embedding (simple approach)
        doc_chunks = [str(el) for el in elements if str(el).strip()] # Filter out empty strings
        if not doc_chunks:
             raise RuntimeError("Document contained no extractable text content after partitioning.")

        # Create IDs and basic metadata
        doc_ids = [f"{collection_name}_{uuid.uuid4()}" for _ in range(len(doc_chunks))] # Use UUIDs for robustness
        metadatas = [{"source": original_filename, "chunk_index": i} for i in range(len(doc_chunks))]

        # --- 4. Add to Vector Store ---
        services.add_document_chunks_service(
            collection_name=collection_name,
            documents=doc_chunks,
            ids=doc_ids,
            metadatas=metadatas
        )

        # --- 5. Create Success Response ---
        response_data = UploadResponse(
            message=f"Document '{original_filename}' processed successfully.",
            collection_name=collection_name,
            chunks_added=len(doc_chunks)
        )
        # Use .dict() in Pydantic v1, .model_dump() in Pydantic v2
        return jsonify(response_data.dict()), 201 # 201 Created

    # No specific except blocks needed here as they are handled by blueprint error handlers
    finally:
        # --- 6. Cleanup Temporary File ---
        if temp_file_path:
            utils.cleanup_file(temp_file_path)


@qa_bp.route('/ask', methods=['POST'])
def ask_question():
    """
    Receives a question and collection name, retrieves context, asks LLM.
    Expects JSON body conforming to AskRequest model.
    """
    try:
        # --- 1. Validate Request Body ---
        # Raises ValidationError if invalid, caught by the error handler
        ask_data = AskRequest(**request.get_json())

        # --- 2. Query Vector Store ---
        # Raises ValueError if collection not found, caught by error handler
        query_results = services.query_vector_store_service(
            collection_name=ask_data.collection_name,
            query_text=ask_data.question,
            n_results=ask_data.n_results,
            where_filter=ask_data.where_filter
        )

        context_docs = query_results.get('documents', [[]])[0] # result is list of lists

        if not context_docs:
            current_app.logger.warning(f"No relevant documents found in '{ask_data.collection_name}' for query: '{ask_data.question}'")
            # Return a specific answer indicating lack of context
            answer = "I couldn't find relevant information in the specified document collection to answer that question."
            context_docs_used = []
        else:
            # --- 3. Prepare Prompt for LLM ---
            context_string = "\n\n".join(context_docs)
            prompt = f"""You are an AI assistant answering questions based ONLY on the provided context below.
If the information is not in the context, say "I don't have enough information in the context to answer that."
Do not make up information or answer based on prior knowledge outside the context.

Context:
---
{context_string}
---

Question: {ask_data.question}

Answer:"""
            messages = [{"role": "user", "content": prompt}]

            # --- 4. Generate Answer using LLM ---
            # Raises ConnectionError or RuntimeError on failure, caught by error handlers
            answer = services.generate_completion_service(messages)
            context_docs_used = context_docs # Record the context used

        # --- 5. Create Success Response ---
        response_data = AskResponse(
            answer=answer,
            question=ask_data.question,
            collection_name=ask_data.collection_name,
            context_used=context_docs_used
        )
        # Use .dict() in Pydantic v1, .model_dump() in Pydantic v2
        return jsonify(response_data.dict()), 200
    
    except Exception as e:
        current_app.logger.exception("Error in /ask endpoint: %s", str(e))
        raise


    # No specific except blocks needed here as they are handled by blueprint error handlers
    # The generic Exception handler will catch anything missed.