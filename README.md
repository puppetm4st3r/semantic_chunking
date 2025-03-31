# Text Processing and Semantic Analysis Toolkit

## Introduction

This repository provides a comprehensive set of tools for natural language processing, text chunking, semantic analysis, and question answering with RAG (Retrieval-Augmented Generation) capabilities. It's designed to process, analyze, and extract meaningful information from text documents, particularly focusing on Spanish-language texts. The toolkit combines classical information retrieval techniques with modern embedding-based semantic search approaches to offer robust text analysis capabilities.

## Repository Structure

- `tools.py`: Core utilities for text processing, keyword extraction, vector database operations, and question answering
- `text_chunking/`: Specialized modules for semantic text chunking and visualization
- `workshop.ipynb`: Demonstration notebook showing the complete workflow 
- `book.txt`: Sample text used in the demonstrations
- `requirements.txt`: Dependencies required for the project

## Core Components (tools.py)

### Text Processing

- `normalize_spanish_text(text)`: Handles Spanish-specific text normalization, removing accents and special characters
- `load_book(path)`: Loads text content from a file
- `format_text_with_line_breaks(text, line_length)`: Formats text with appropriate line breaks

### Keyword Extraction and Visualization

- `extract_keywords_with_bm25(fragment_and_full_text)`: Extracts relevant keywords from text fragments using BM25 algorithm
- `extract_keywords_from_fragments(fragments, full_text, top_n, max_workers)`: Handles parallel keyword extraction from multiple text fragments
- `generate_wordcloud(keywords_results, title, output_dir, ...)`: Creates visual wordcloud representations of extracted keywords

### Vector Database Operations

- `connect_to_postgres(host, port, dbname, user, password)`: Establishes connection to PostgreSQL database
- `check_chunks_table(connection, vector_dimensions)`: Ensures proper table structure for vector storage
- `get_embedding(text, api_key, base_url_embeddings, model)`: Obtains vector embeddings for text
- `hybrid_search(connection, query_text, api_key, ...)`: Performs combined semantic and keyword-based search
- `initialize_vector_database(host, port, dbname, user, password, vector_dimensions)`: Sets up vector database with pgvector extension
- `insert_text_fragments(connection, text_fragments, keywords_lists, api_key, ...)`: Stores text fragments with their embeddings and keywords

### Retrieval and Question Answering

- `reciprocal_rank_fusion(results, top_k, k_constant)`: Reranks search results using RRF algorithm
- `get_openai_response(prompt, api_key, base_url, model, temperature)`: Obtains responses from OpenAI models
- `answer_question_with_context(query, context_docs, api_key, base_url, model, prompt_template)`: Generates answers based on retrieved context
- `answer_question_with_context_streaming(query, context_docs, api_key, base_url, model, prompt_template)`: Stream-based version of answer generation
- `process_text_into_chunks(fulltext)`: Divides text into appropriate chunks for processing

## Text Chunking Module (text_chunking/)

### SemanticSplitGenerator (text_chunking/splitting/SemanticSplitGenerator.py)

This class is responsible for semantically aware text splitting:

- `__init__(llm_chain, split_texts, split_text_embeddings)`: Initializes with text and embedding data
- `build_chunk_cosine_distances()`: Calculates similarity between consecutive chunks
- `get_breakpoints(embeddings, start, end, threshold)`: Identifies semantic breakpoints in text
- `build_chunks_stack(length_threshold, cosine_distance_percentile_threshold)`: Creates text chunks respecting semantic boundaries
- `build_semantic_groups(breakpoints)`: Groups text splits based on semantic similarity
- `build_semantic_group_clusters(semantic_groups, cluster_ids)`: Aggregates semantic groups into clusters
- `build_semantic_group_summaries(semantic_groups_to_summarize, verbose)`: Generates summaries for semantic groups

### SemanticClusterVisualizer (text_chunking/SemanticClusterVisualizer.py)

This class provides visualization and management capabilities for semantic clusters:

- `__init__(api_key, llm_model, temperature, base_url_llm, base_url_embeddings, embeddings_model)`: Initializes visualization capabilities
- `split_documents(splitter, documents, min_chunk_len, verbose)`: Splits documents and handles short chunks
- `merge_short_documents(split_texts, min_len)`: Combines small text fragments for better analysis
- `embed_original_document_splits(doc_splits)`: Creates embeddings for document chunks
- `embed_semantic_groups(semantic_groups)`: Creates embeddings for semantic groups
- `generate_breakpoints(doc_splits, doc_split_embeddings, length_threshold, percentile_threshold, plot, verbose)`: Creates semantic breakpoints for document organization
- `vizualize_semantic_groups(semantic_groups, semantic_group_embeddings, n_clusters)`: Visualizes semantic relationships between text groups
- `generate_cluster_labels(semantic_group_clusters, plot)`: Creates labels and summaries for semantic clusters

## Workshop Notebook (workshop.ipynb)

The notebook demonstrates a complete workflow for text processing and RAG-based question answering:

1. **Database Setup**: Initializes a PostgreSQL database with vector capabilities
2. **Text Ingestion**: Loads and performs initial processing of text
3. **Semantic Processing**: Creates embeddings and performs semantic chunking
4. **Keyword Extraction**: Identifies important terms in each text fragment
5. **Hybrid Index Creation**: Builds combined semantic and keyword-based search capabilities
6. **Hybrid Retrieval**: Performs searches using both semantic and keyword methods
7. **Relevance Ranking**: Reorganizes results by relevance using Reciprocal Rank Fusion
8. **Response Generation**: Creates natural language answers based on retrieved context

## Usage Examples

The `workshop.ipynb` notebook provides step-by-step examples of using the toolkit, including:

1. Setting up a vector database
2. Loading and processing text
3. Extracting keywords and generating visualizations
4. Performing semantic analysis and clustering
5. Building hybrid search capabilities
6. Asking questions and generating contextualized answers

## Requirements

Key dependencies include:

- Python 3.10+
- PostgreSQL with pgvector extension
- OpenAI API or compatible local models
- Various Python libraries (numpy, scipy, langchain, etc.)

See `requirements.txt` for a complete list of dependencies.

## Deployment Options

### Using OpenAI API

The toolkit is configured to work with the OpenAI API by default. To use OpenAI's services:

1. Obtain an API key from OpenAI
2. In the notebook, make sure the `base_url` parameters point to the OpenAI API endpoints:
   ```python
   semantic_chunker = SemanticClusterVisualizer(
       api_key="your-openai-api-key", 
       llm_model='gpt-4o',
       base_url_llm="https://api.openai.com/v1",
       base_url_embeddings="https://api.openai.com/v1",
       embeddings_model="text-embedding-3-small"
   )
   ```

### Using Local LLM with vLLM

For privacy, cost efficiency, or customization, you can run models locally using [vLLM](https://github.com/vllm-project/vllm) as an OpenAI-compatible API backend:

1. Install vLLM:
   ```bash
   pip install vllm
   ```

2. Start the vLLM server with an OpenAI-compatible API:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model your-local-model-name \
       --host 127.0.0.1 \
       --port 3000
   ```

3. Launch an embedding model server on a different port:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model your-embedding-model-name \
       --host 127.0.0.1 \
       --port 3001
   ```

4. In the notebook, simply change the API URLs to point to your local servers:
   ```python
   semantic_chunker = SemanticClusterVisualizer(
       api_key="123",  # Any string works when using vLLM locally
       llm_model='gpt-4o',
       base_url_llm="http://localhost:3000/v1",
       base_url_embeddings="http://localhost:3001/v1",
       embeddings_model="text-embedding-3-small"
   )
   ```

That's it! The toolkit will now use your local LLM infrastructure instead of the OpenAI API, while maintaining the same functionality. No other code changes are required beyond updating the hostnames and API keys in the notebook.

## Configuration Points in the Notebook

When using the notebook, the following points need to be modified according to your specific setup:

1. **Database Connection Parameters** (Cell 3):
   ```python
   connection = initialize_vector_database(
       host="localhost",  # Change to your Postgres server
       port=5432,         # Change if using a different port
       dbname="workshop_rag",  # Your database name
       user="postgres",   # Your database username
       password="dev.2m", # Your database password
       vector_dimensions=1024
   )
   ```

2. **LLM and Embedding Services Configuration** (Cell 4):
   ```python
   semantic_chunker = SemanticClusterVisualizer(
       api_key="123",  # Your API key for OpenAI or local service
       llm_model='gpt-4o',  # Change to your preferred model
       base_url_llm="http://localhost:3000/v1",  # URL for LLM service
       base_url_embeddings="http://localhost:3001/v1",  # URL for embeddings service
       embeddings_model="text-embedding-3-small"  # Change to your preferred embedding model
   )
   ```

3. **Hybrid Search Parameters** (Cell 10):
   ```python
   results = hybrid_search(
       connection=connection,
       query_text=query,
       api_key="123",  # Your API key
       base_url_embeddings="http://localhost:3001/v1",  # URL for embeddings service
       top_k=20
   )
   ```

4. **Question Answering Service Configuration** (Cell 12):
   ```python
   stream = answer_question_with_context_streaming(
       query=query, 
       context_docs=results, 
       api_key="123",  # Your API key
       base_url="http://localhost:3000/v1",  # URL for LLM service
       model="gpt-4o",  # Change to your preferred model
       prompt_template=prompt
   )
   ```

These are all the configuration points that need to be modified in the notebook to adapt it to your environment, whether you're using OpenAI's services or a local setup with vLLM.

## Credits and Acknowledgements

The semantic chunking and visualization functionality in this repository is based on the excellent work by [rmartinshort](https://github.com/rmartinshort) in the [text_chunking](https://github.com/rmartinshort/text_chunking) repository. We've extended and adapted these core mechanisms to handle Spanish text, implement hybrid search capabilities, and integrate them with a RAG system. We are grateful for the original implementation that provided a strong foundation for semantic text processing.

## Copyright and License

Copyright © 2025 Dolfs SpA (https://www.dolfs.io)

This project is licensed under the MIT License. 
