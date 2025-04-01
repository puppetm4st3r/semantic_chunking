# Text Processing and Semantic Analysis Toolkit

## Introduction

This repository provides a comprehensive set of tools for natural language processing, text chunking, semantic analysis, and question answering with RAG (Retrieval-Augmented Generation) capabilities. It's designed to process, analyze, and extract meaningful information from text documents, particularly focusing on Spanish-language texts. The toolkit implements a sophisticated hybrid search system that combines traditional full-text search with modern vector-based semantic search, delivering highly relevant results for complex queries.

### Technical Architecture

The core functionality in `tools.py` implements a complete RAG (Retrieval-Augmented Generation) pipeline with the following detailed components:

#### 1. Text Processing and Normalization
The system includes specialized functions for handling Spanish text, such as `normalize_spanish_text()`, which removes accents and normalizes special characters through character mapping. This ensures consistent text representation regardless of accent variations common in Spanish language texts. The text processing pipeline also includes functions like `format_text_with_line_breaks()` for proper text formatting and `process_text_into_chunks()` which uses regex patterns and language structure rules to intelligently divide text into semantically coherent units.

#### 2. Semantic Text Chunking
Text is broken into semantically meaningful chunks through a multi-stage process:
- Initial chunking using paragraph and sentence boundaries
- Refinement using semantic similarity thresholds, specifically detecting significant content shifts at the 95th percentile of cosine distance between adjacent chunk embeddings
- The `SemanticSplitGenerator.get_breakpoints()` method identifies these semantic boundaries by analyzing embedding distance patterns
- The system identifies natural semantic breakpoints where the cosine distance between embeddings exceeds the threshold, indicating a topic or content transition
- Further semantic analysis using embedding-based clustering to identify thematic relationships
- The `process_text_into_chunks()` function implements a sophisticated algorithm that joins lines where the next line doesn't start with an uppercase letter, splits by double newlines, and further divides text where periods are followed by uppercase letters to preserve logical document structure.

#### 3. Keyword Extraction with BM25
The system employs the Okapi BM25 ranking algorithm (implemented in `extract_keywords_with_bm25()`) to identify the most relevant keywords from each text fragment:
- For each text fragment, the function tokenizes and cleans both the fragment and the full text
- Creates a corpus from paragraphs or sentences in the full text
- Applies BM25 scoring to calculate the relevance of each term in the fragment against the full corpus
- Filters Spanish stopwords using NLTK or a fallback list when NLTK is unavailable
- Parallelizes keyword extraction for efficiency using Python's `concurrent.futures` with configurable worker limits

#### 4. Vector Database Implementation
The system uses PostgreSQL with the pgvector extension to store and retrieve vector embeddings:
- `initialize_vector_database()` sets up PostgreSQL with pgvector extension and creates the necessary tables
- `check_chunks_table()` ensures proper table structure with vector dimensions matching the embedding model
- The database schema includes dedicated columns for:
  - Raw text content (`content`)
  - Vector embeddings (`content_vector` using pgvector's vector type)
  - Text search index (`content_tsvector` as a generated column using PostgreSQL's `to_tsvector('spanish', content)`)
  - Metadata in JSONB format (storing keywords, timestamps, etc.)
- Creates optimal indices for both vector and text search:
  - HNSW (Hierarchical Navigable Small World) index for efficient vector search with configurable parameters (`ef_construction = 200, m = 16`)
  - GIN index for PostgreSQL's full-text search on the tsvector column

#### 5. Hybrid Search Implementation
The `hybrid_search()` function implements a sophisticated dual-retrieval approach:
- **Vector Similarity Search**:
  - Converts query text to vector embedding using OpenAI-compatible models
  - Uses pgvector's vector distance operator (`<=>`) to find semantically similar documents
  - Calculates a normalized similarity score (0-100) based on vector cosine distance
  
- **Full-Text Keyword Search**:
  - Extracts keywords from the query after removing Spanish stopwords
  - Uses PostgreSQL's `websearch_to_tsquery('spanish', keywords)` to create a text search query
  - Employs `ts_rank_cd` with custom weight configurations `'{0.1, 0.2, 0.4, 1.0}'` to prioritize document sections
  - Boosts scores based on exact keyword matches with context-aware weighting

#### 6. Result Fusion and Ranking
The `reciprocal_rank_fusion()` function implements a modified version of the Reciprocal Rank Fusion (RRF) algorithm:
- Combines results from both vector and text searches using a weighted approach
- For each document, calculates a fusion score using the formula: `1/(k + rank)` where k is a constant (default: 60)
- Weights the RRF components based on the original relevance scores from each method
- Applies a 20% boost to documents found by both search methods
- Normalizes final scores to a 0-100 range for consistency
- Returns a combined list of results sorted by the fused relevance score

The algorithm specifically:
1. Processes vector and text search results separately, calculating RRF components for each
2. Assigns weights based on original relevance scores to preserve quality signals
3. Incorporates both rank position and score magnitude in the final fusion formula
4. Accounts for the "methods_count" to give preference to documents found through multiple search methods
5. Produces a unified, re-ranked list of the most relevant context documents

#### 7. Answer Generation
The system implements two approaches for answer generation:
- `answer_question_with_context()`: Standard synchronous response generation
- `answer_question_with_context_streaming()`: Token-by-token streaming for real-time responses

Both functions:
- Format retrieved context documents with sequential numbering
- Construct a prompt that includes the query and retrieved context
- Call the language model (OpenAI API or compatible local model)
- Apply appropriate system instructions to generate concise, accurate answers

### Deployment Flexibility

The implementation is designed to work with either OpenAI's API or local language models through compatible interfaces like vLLM, making it flexible for various deployment scenarios. For local deployments, the system uses identical API signatures but points to local endpoints, ensuring a consistent interface regardless of the underlying model provider.

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

With ❤️ from Dolfs SpA (https://www.dolfs.io)

This project is licensed under the MIT License.
