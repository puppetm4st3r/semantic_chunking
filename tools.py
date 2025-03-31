import re
import numpy as np
from nltk.corpus import stopwords
import nltk
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import concurrent.futures
import multiprocessing
import psycopg2
import psycopg2.extras
import openai
import json
import datetime
from langchain_openai import OpenAIEmbeddings

# Download NLTK resources once at module level
try:
    nltk.download('stopwords', quiet=True)
    # Get Spanish stopwords
    SPANISH_STOPWORDS = set(stopwords.words('spanish'))
except Exception as e:
    print("Error loading Spanish stopwords. Falling back to a basic list.")
    # Fallback basic Spanish stopwords list
    SPANISH_STOPWORDS = set(['a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra', 
                        'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella', 'ellas', 
                        'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres', 'es', 'esa', 'esas', 
                        'ese', 'eso', 'esos', 'esta', 'estaba', 'estabais', 'estaban', 'estabas', 'estad', 
                        'estada', 'estadas', 'estado', 'estados', 'estamos', 'estando', 'estar', 'estaremos', 
                        'estará', 'estarán', 'estarás', 'estaré', 'estaréis', 'estaría', 'estaríais', 
                        'estaríamos', 'estarían', 'estarías', 'estas', 'este', 'estemos', 'esto', 'estos', 
                        'estoy', 'estuve', 'estuviera', 'estuvierais', 'estuvieran', 'estuvieras', 'estuvieron', 
                        'estuviese', 'estuvieseis', 'estuviesen', 'estuvieses', 'estuvimos', 'estuviste', 
                        'estuvisteis', 'estuviéramos', 'estuviésemos', 'hubo', 'fue', 'fuera', 'fuerais', 
                        'fueran', 'fueras', 'fueron', 'fuese', 'fueseis', 'fuesen', 'fueses', 'fui', 'fuimos', 
                        'fuiste', 'fuisteis', 'fuéramos', 'fuésemos', 'ha', 'habida', 'habidas', 'habido', 
                        'habidos', 'habiendo', 'habremos', 'habrá', 'habrán', 'habrás', 'habré', 'habréis', 
                        'habría', 'habríais', 'habríamos', 'habrían', 'habrías', 'han', 'has', 'hasta', 'hay', 
                        'haya', 'hayamos', 'hayan', 'hayas', 'hayáis', 'he', 'hemos', 'hube', 'hubiera', 
                        'hubierais', 'hubieran', 'hubieras', 'hubieron', 'hubiese', 'hubieseis', 'hubiesen', 
                        'hubieses', 'hubimos', 'hubiste', 'hubisteis', 'hubiéramos', 'hubiésemos', 'hubo', 
                        'la', 'las', 'le', 'les', 'lo', 'los', 'me', 'mi', 'mis', 'mucho', 'muchos', 'muy', 
                        'más', 'mí', 'mía', 'mías', 'mío', 'míos', 'nada', 'ni', 'no', 'nos', 'nosotras', 
                        'nosotros', 'nuestra', 'nuestras', 'nuestro', 'nuestros', 'o', 'os', 'otra', 'otras', 
                        'otro', 'otros', 'para', 'pero', 'poco', 'por', 'porque', 'que', 'quien', 'quienes', 
                        'qué', 'se', 'sea', 'seamos', 'sean', 'seas', 'ser', 'seremos', 'será', 'serán', 
                        'serás', 'seré', 'seréis', 'sería', 'seríais', 'seríamos', 'serían', 'serías', 'seáis', 
                        'si', 'sido', 'siendo', 'sin', 'sobre', 'sois', 'somos', 'son', 'soy', 'su', 'sus', 
                        'suya', 'suyas', 'suyo', 'suyos', 'sí', 'también', 'tanto', 'te', 'tendremos', 'tendrá', 
                        'tendrán', 'tendrás', 'tendré', 'tendréis', 'tendría', 'tendríais', 'tendríamos', 
                        'tendrían', 'tendrías', 'tened', 'tenemos', 'tenga', 'tengamos', 'tengan', 'tengas', 
                        'tengo', 'tengáis', 'tenida', 'tenidas', 'tenido', 'tenidos', 'teniendo', 'tenéis', 
                        'tenía', 'teníais', 'teníamos', 'tenían', 'tenías', 'ti', 'tiene', 'tienen', 'tienes', 
                        'todo', 'todos', 'tu', 'tus', 'tuve', 'tuviera', 'tuvierais', 'tuvieran', 'tuvieras', 
                        'tuvieron', 'tuviese', 'tuvieseis', 'tuviesen', 'tuvieses', 'tuvimos', 'tuviste', 
                        'tuvisteis', 'tuviéramos', 'tuviésemos', 'tuvo', 'tuya', 'tuyas', 'tuyo', 'tuyos', 
                        'tú', 'un', 'una', 'uno', 'unos', 'vosotras', 'vosotros', 'vuestra', 'vuestras', 
                        'vuestro', 'vuestros', 'y', 'ya', 'yo', 'él', 'éramos'])

# Spanish specific cleanup for accents and special characters
def normalize_spanish_text(text: str) -> str:
    # Map for accents and special characters
    accents_map = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n', 'à': 'a', 'è': 'e', 'ì': 'i',
        'ò': 'o', 'ù': 'u'
    }
    
    # Replace accented characters
    for accented, normal in accents_map.items():
        text = text.replace(accented, normal)
        
    return text

def load_book(path: str) -> str:
    """Load the content of a text file and return it as a string."""
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()  # Read the entire content of the file
    return content  # Return the content of the file

def format_text_with_line_breaks(text: str, line_length: int = 100) -> str:
    """
    Formats text with line breaks every specified number of characters,
    preserving existing line breaks and resetting the counter after them.
    
    Args:
        text: The text to format
        line_length: Maximum length of each line
        
    Returns:
        Formatted text with appropriate line breaks
    """
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if len(line) <= line_length:
            formatted_lines.append(line)
        else:
            current_line = ""
            words = line.split(' ')
            
            for word in words:
                if len(current_line) + len(word) + 1 <= line_length:
                    if current_line:
                        current_line += ' ' + word
                    else:
                        current_line = word
                else:
                    formatted_lines.append(current_line)
                    current_line = word
            
            if current_line:
                formatted_lines.append(current_line)
    
    return '\n'.join(formatted_lines)

def extract_keywords_with_bm25(fragment_and_full_text: tuple) -> list[str]:
    """
    Extract keywords from a fragment of text that are relevant within the full text using BM25.
    Uses NLTK to remove Spanish stopwords.
    
    Args:
        fragment_and_full_text: Tuple containing (fragment, full_text, top_n)
        
    Returns:
        List of relevant keywords
    """
    fragment, full_text, top_n = fragment_and_full_text
    
    # Tokenize and clean the full text using simple split instead of word_tokenize
    def tokenize_and_clean(text: str) -> list[str]:
        # Normalize text for Spanish
        text = normalize_spanish_text(text.lower())
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Simple tokenization by splitting on whitespace
        words = text.split()
        # Remove stopwords and short words (less than 3 characters)
        filtered_words = [word for word in words if word not in SPANISH_STOPWORDS and len(word) > 2]
        return filtered_words
    
    # Split full text into paragraphs or sentences for corpus
    paragraphs = [p for p in re.split(r'\n+', full_text) if p.strip()]
    if len(paragraphs) < 5:  # If not enough paragraphs, use sentences
        paragraphs = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
    
    # Tokenize paragraphs for BM25
    tokenized_corpus = [tokenize_and_clean(p) for p in paragraphs]
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize fragment
    tokenized_fragment = tokenize_and_clean(fragment)
    
    # Get unique words from fragment (candidate keywords)
    unique_words = list(set(tokenized_fragment))
    
    # Calculate scores for each word
    word_scores = {}
    for word in unique_words:
        # Calculate BM25 score for the word across all paragraphs
        doc_scores = np.array([bm25.get_scores([word])[i] for i in range(len(paragraphs))])
        word_scores[word] = np.mean(doc_scores)  # Average score across all paragraphs
    
    # Sort words by score in descending order
    sorted_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return [word for word, score in sorted_keywords[:top_n]]

def extract_keywords_from_fragments(fragments: list[str], full_text: str, top_n: int = 10, max_workers: int = None) -> list[list[str]]:
    """
    Extract keywords from multiple text fragments using BM25 in parallel.
    
    Args:
        fragments: List of text fragments to extract keywords from
        full_text: The complete text for context
        top_n: Number of top keywords to return for each fragment
        max_workers: Maximum number of parallel workers (defaults to number of CPU cores)
        
    Returns:
        List of lists, where each inner list contains keywords for the corresponding fragment
    """
    # If max_workers is not specified, use all available cores
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Prepare the input data for parallel processing
    input_data = [(fragment, full_text, top_n) for fragment in fragments]
    results = [None] * len(fragments)  # Pre-allocate results list
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a mapping of futures to their indices
        future_to_index = {executor.submit(extract_keywords_with_bm25, data): i 
                            for i, data in enumerate(input_data)}
        
        # Process results as they complete using tqdm for progress tracking
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                        total=len(fragments), 
                        desc=f"Extracting keywords (using {max_workers} cores)"):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Error processing fragment {index}: {e}")
                results[index] = []  # Empty list in case of error
    
    return results

def generate_wordcloud(keywords_results: list[list[str]], 
                        title: str,
                        output_dir: str = None,
                        width: int = 1200, 
                        height: int = 600, 
                        combine_all: bool = True,
                        background_color: str = 'black',
                        max_words: int = 500,
                        colormap: str = 'viridis',
                        ) -> None:
    """
    Generate wordcloud visualizations from keywords extraction results.
    By default, combines all keywords into a single wordcloud.
    
    Args:
        keywords_results: List of keyword lists from extract_keywords_from_fragments
        title: Title for the wordcloud
        output_dir: Directory to save the generated wordcloud images (optional)
        width: Width of the wordcloud image
        height: Height of the wordcloud image
        combine_all: If True (default), generates a single wordcloud for all keywords combined
        background_color: Background color of the wordcloud
        max_words: Maximum number of words to include in the wordcloud
        colormap: Color map to use for the wordcloud (good options for dark backgrounds: 
                'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'rainbow')
        
    Returns:
        None (displays or saves the wordcloud visualizations)
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define color function for bright colors on dark background
    def get_bright_color_func(colormap_name):
        from matplotlib.colors import LinearSegmentedColormap
        import random
        
        # Predefined color maps optimized for dark backgrounds
        if colormap_name == 'neon':
            # Custom neon colors for dark background
            colors = ['#ff00ff', '#00ffff', '#ff0000', '#00ff00', '#0000ff', 
                    '#ffff00', '#ff8000', '#00ff80', '#8000ff']
            cmap = LinearSegmentedColormap.from_list('neon', colors, N=len(colors))
        else:
            # Use matplotlib colormap
            cmap = plt.cm.get_cmap(colormap_name)
        
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if random_state is None:
                random_state = random.Random()
            # Get a color from the colormap (brighten for dark backgrounds)
            r, g, b, _ = cmap(random_state.random())
            # Make colors brighter for dark backgrounds
            r = min(1.0, r * 1.5)
            g = min(1.0, g * 1.5)
            b = min(1.0, b * 1.5)
            return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            
        return color_func
    
    # Function to create a wordcloud from a list of keywords
    def create_wordcloud(keywords: list[str], cloud_title: str, filename: str = None):
        # Create frequency dictionary where each keyword has a value of its position in reverse
        # This gives higher weight to words that appear earlier in the list (more relevant)
        word_freq = {}
        for i, word in enumerate(keywords):
            weight = len(keywords) - i
            if word in word_freq:
                word_freq[word] += weight
            else:
                word_freq[word] = weight
        
        # Generate wordcloud with color function
        color_func = get_bright_color_func(colormap)
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            color_func=color_func,
            prefer_horizontal=0.9,
            relative_scaling=0.7,
            min_font_size=8,
            collocations=False  # Avoid repeating similar words
        ).generate_from_frequencies(word_freq)
        
        # Display the wordcloud
        plt.figure(figsize=(width/100, height/100), facecolor=background_color)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(cloud_title, color='white' if background_color == 'black' else 'black')
        plt.tight_layout(pad=0)
        
        # Save if output directory is specified
        if output_dir and filename:
            plt.savefig(os.path.join(output_dir, filename), 
                        bbox_inches='tight', dpi=150, 
                        facecolor=background_color)
            plt.close()
        else:
            plt.show()
    
    # Generate combined wordcloud (default behavior)
    if combine_all:
        # Create a frequency dictionary with weighted scores
        word_freq = {}
        for group_idx, keywords in enumerate(keywords_results):
            for i, word in enumerate(keywords):
                weight = len(keywords) - i  # Higher weight for more relevant words
                if word in word_freq:
                    word_freq[word] += weight
                else:
                    word_freq[word] = weight
        
        # Generate and display/save the combined wordcloud
        filename = "combined_wordcloud.png" if output_dir else None
        
        # Generate wordcloud from frequencies with color function
        color_func = get_bright_color_func(colormap)
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            color_func=color_func,
            prefer_horizontal=0.9,
            relative_scaling=0.7,
            min_font_size=8,
            collocations=False  # Avoid repeating similar words
        ).generate_from_frequencies(word_freq)
        
        # Display the wordcloud
        plt.figure(figsize=(width/100, height/100), facecolor=background_color)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, color='white' if background_color == 'black' else 'black')
        plt.tight_layout(pad=0)
        
        # Save if output directory is specified
        if output_dir and filename:
            plt.savefig(os.path.join(output_dir, filename), 
                        bbox_inches='tight', dpi=150, 
                        facecolor=background_color)
            plt.close()
        else:
            plt.show()
    
    # Generate individual wordclouds for each fragment
    else:
        for i, keywords in enumerate(keywords_results):
            segment_title = f"Fragment {i+1} Keywords"
            filename = f"fragment_{i+1}_wordcloud.png" if output_dir else None
            create_wordcloud(keywords, segment_title, filename)

def connect_to_postgres(host, port, dbname, user, password):
    """
    Connect to PostgreSQL database and return the connection.
    
    Args:
        host: Database host address
        port: Database port
        dbname: Database name
        user: Database user
        password: Database password
        
    Returns:
        Connection object to the PostgreSQL database
    """
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user, 
            password=password
        )
        return connection
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        raise e

def check_chunks_table(connection, vector_dimensions=1024):
    """
    Check if the chunks table exists with the necessary structure for vector and text search.
    Creates the table with appropriate indices if it doesn't exist.
    
    Args:
        connection: PostgreSQL database connection
        vector_dimensions: Dimensions of the vector to store (defaults to 1024 for text-embedding-3-small)
        
    Returns:
        Boolean indicating whether the table was created (True) or already existed (False)
    """
    cursor = connection.cursor()
    created = False
    
    try:
        # Check if pgvector extension is installed
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone() is None:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            connection.commit()
            print("Created pgvector extension")
        
        # Check if chunks table exists
        cursor.execute("SELECT to_regclass('public.chunks');")
        table_exists = cursor.fetchone()[0] is not None
        
        if not table_exists:
            # Create chunks table with appropriate columns for vector and text search
            cursor.execute(f"""
                CREATE TABLE chunks (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_vector vector({vector_dimensions}),
                    content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED,
                    metadata JSONB
                );
            """)
            
            # Create indices for efficient search
            # Use HNSW index for better vector search performance
            cursor.execute("""
                CREATE INDEX chunks_content_vector_idx ON chunks 
                USING hnsw (content_vector vector_cosine_ops) 
                WITH (ef_construction = 200, m = 16);
            """)
            cursor.execute("CREATE INDEX chunks_content_tsvector_idx ON chunks USING GIN (content_tsvector);")
            
            connection.commit()
            created = True
            print(f"Created chunks table with vector({vector_dimensions}) and HNSW index for vector search")
        
    except Exception as e:
        connection.rollback()
        print(f"Error checking/creating chunks table: {e}")
        raise e
    finally:
        cursor.close()
    
    return created

def get_embedding(text, api_key, base_url_embeddings=None, model="text-embedding-3-small"):
    """
    Get embeddings for a text using langchain's OpenAIEmbeddings.
    
    Args:
        text: The text to get embeddings for
        api_key: OpenAI API key for authentication
        base_url_embeddings: Base URL for the OpenAI API (optional)
        model: The OpenAI embedding model to use
        
    Returns:
        The vector representation of the text
    """
    # Ensure text is not empty
    if not text or text.strip() == "":
        text = "empty"
    
    # Initialize OpenAIEmbeddings from langchain
    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url_embeddings
    )
    
    # Get the embedding vector
    embedding = embeddings.embed_documents([text])[0]
    return embedding

def hybrid_search(connection, query_text, api_key, base_url_embeddings=None, model="text-embedding-3-small", top_k=10):
    """
    Perform hybrid search combining vector similarity and keyword text search.
    
    Args:
        connection: PostgreSQL database connection
        query_text: Text query for both semantic and full-text search
        api_key: OpenAI API key for authentication
        base_url_embeddings: Base URL for the OpenAI API (optional)
        model: The OpenAI embedding model to use
        top_k: Total number of results to return
        
    Returns:
        List of dictionaries containing search results with normalized scores
    """
    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
    half_k = max(1, top_k // 2)
    results = []
    
    try:
        # Count total rows in the chunks table first for diagnostics
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        print(f"Total chunks in database: {total_chunks}")
        
        # Extract keywords from query_text
        # Normalize text for Spanish
        query_normalized = normalize_spanish_text(query_text.lower())
        # Remove punctuation and special characters
        query_normalized = re.sub(r'[^\w\s]', '', query_normalized)
        # Split into words
        words = query_normalized.split()
        # Filter out stopwords and short words
        keywords = [word for word in words if word not in SPANISH_STOPWORDS and len(word) > 2]
        
        # If no keywords extracted, use the most significant words from the query
        if not keywords and words:
            # Use words even if they are stopwords or short, just take the longest ones
            keywords = sorted(words, key=len, reverse=True)[:3]  # Take up to 3 longest words
        
        print(f"Extracted keywords: {keywords}")
        
        # Generate embedding vector from the query text
        query_vector = get_embedding(query_text, api_key, base_url_embeddings, model)
        print(f"Generated query vector with {len(query_vector)} dimensions")
        
        # Part 1: Vector similarity search (semantic)
        vector_query = """
            SELECT 
                id, 
                content, 
                metadata,
                1 - (content_vector <=> %s::vector) AS similarity
            FROM 
                chunks
            ORDER BY 
                content_vector <=> %s::vector
            LIMIT %s;
        """
        
        cursor.execute(vector_query, (query_vector, query_vector, half_k))
        vector_results = cursor.fetchall()
        print(f"Vector search returned {len(vector_results)} results")
        
        # Normalize vector scores to 0-100
        for row in vector_results:
            score = float(row['similarity']) * 100  # Convert to 0-100 scale
            results.append({
                'id': row['id'],
                'content': row['content'],
                'metadata': row['metadata'],
                'score': score,
                'search_type': 'vector'
            })
        
        # Part 2: Keyword text search - only if we have keywords
        if keywords and len(keywords) > 0:
            try:
                # Simple version using a single plainto_tsquery with all keywords
                single_query = """
                    SELECT
                        id,
                        content,
                        metadata,
                        ts_rank_cd('{0.1, 0.2, 0.4, 1.0}', content_tsvector, websearch_to_tsquery('spanish', %s)) AS text_rank,
                        content_tsvector
                    FROM
                        chunks
                    WHERE
                        content_tsvector @@ websearch_to_tsquery('spanish', %s)
                    ORDER BY
                        text_rank DESC
                    LIMIT %s;
                """
                
                # Join all keywords with spaces to use in a single query
                combined_keywords = ' OR '.join(keywords)
                
                print(f"Combined keywords for search: '{combined_keywords}'")
                
                # Execute with the combined keywords
                cursor.execute(single_query, (combined_keywords, combined_keywords, half_k))
                text_results = cursor.fetchall()
                print(f"Text search returned {len(text_results)} results")
                
                # Find max rank to normalize scores
                max_rank = 0
                for row in text_results:
                    max_rank = max(max_rank, float(row['text_rank']))
                
                # Add text search results with normalized scores
                for row in text_results:
                    # Normalize to 0-100 scale
                    norm_score = (float(row['text_rank']) / max_rank * 100) if max_rank > 0 else 0
                    
                    # Count keyword matches to adjust relevance score
                    keyword_match_count = 0
                    content_lower = row['content'].lower()
                    for keyword in keywords:
                        if keyword.lower() in content_lower:
                            keyword_match_count += 1
                    
                    # Boost score based on keyword matches (more matches = higher boost)
                    keyword_boost = (keyword_match_count / len(keywords)) * 20  # Up to 20% boost
                    final_score = min(100, norm_score + keyword_boost)  # Cap at 100
                    
                    results.append({
                        'id': row['id'],
                        'content': row['content'],
                        'metadata': row['metadata'],
                        'score': final_score,
                        'search_type': 'text',
                        'matched_keywords': keyword_match_count
                    })
            except Exception as e:
                print(f"Error in keyword search: {e}")
                # Continue with only vector results if keyword search fails
                pass
        else:
            print("No keywords extracted for text search, using only vector search results")
            # If no keywords, increase the number of vector results to compensate
            if half_k < top_k and len(results) > 0:
                try:
                    # Get existing IDs
                    existing_ids = [r['id'] for r in results]
                    
                    if existing_ids:
                        # Construct a safe query with exact number of parameters
                        id_placeholders = ','.join(['%s' for _ in existing_ids])
                        additional_query = f"""
                            SELECT 
                                id, 
                                content, 
                                metadata,
                                1 - (content_vector <=> %s::vector) AS similarity
                            FROM 
                                chunks
                            WHERE 
                                id NOT IN ({id_placeholders})
                            ORDER BY 
                                content_vector <=> %s::vector
                            LIMIT %s;
                        """
                        
                        # Construct parameters list carefully
                        params = [query_vector] + existing_ids + [query_vector, top_k - len(results)]
                        
                        cursor.execute(additional_query, params)
                        additional_results = cursor.fetchall()
                        
                        # Add additional vector results
                        for row in additional_results:
                            score = float(row['similarity']) * 100
                            results.append({
                                'id': row['id'],
                                'content': row['content'],
                                'metadata': row['metadata'],
                                'score': score,
                                'search_type': 'vector'
                            })
                    else:
                        # If no existing results, just get more vector results
                        cursor.execute(vector_query, (query_vector, query_vector, top_k))
                        additional_results = cursor.fetchall()
                        
                        # Add additional vector results
                        for row in additional_results:
                            score = float(row['similarity']) * 100
                            results.append({
                                'id': row['id'],
                                'content': row['content'],
                                'metadata': row['metadata'],
                                'score': score,
                                'search_type': 'vector'
                            })
                except Exception as e:
                    print(f"Error fetching additional vector results: {e}")
        
        # Remove duplicates (keep highest score)
        unique_results = {}
        for item in results:
            item_id = item['id']
            if item_id not in unique_results:
                unique_results[item_id] = item
            else:
                # If this is a duplicate, update the search_type to "text+vector"
                if unique_results[item_id]['score'] < item['score']:
                    # Keep the higher score item but change search_type
                    item['search_type'] = "text+vector"
                    unique_results[item_id] = item
                else:
                    # Keep the existing item but change its search_type
                    unique_results[item_id]['search_type'] = "text+vector"
        
        # Sort by score in descending order (without limiting to top_k)
        final_results = sorted(list(unique_results.values()), key=lambda x: x['score'], reverse=True)
        
        print(f"Final results count: {len(final_results)}")
        if not final_results:
            # If no results at all, try a desperate measure: get any chunks without filtering
            try:
                cursor.execute("SELECT id, content, metadata FROM chunks LIMIT 5")
                raw_results = cursor.fetchall()
                print(f"Desperate measure returned {len(raw_results)} results")
                for row in raw_results:
                    final_results.append({
                        'id': row['id'],
                        'content': row['content'],
                        'metadata': row['metadata'],
                        'score': 0,
                        'search_type': 'fallback'
                    })
            except Exception as e:
                print(f"Even desperate measure failed: {e}")
        
        return final_results
        
    except Exception as e:
        print(f"Error performing hybrid search: {e}")
        raise e
    finally:
        cursor.close()

def initialize_vector_database(host, port, dbname, user, password, vector_dimensions=1024):
    """
    Initialize vector database by checking if the database exists,
    creating it if needed, and activating the pgvector extension.
    
    Args:
        host: Database host address
        port: Database port
        dbname: Database name to check/create
        user: Database user (must have permissions to create databases)
        password: Database password
        vector_dimensions: Dimensions of the vector to store (defaults to 1024 for text-embedding-3-small)
        
    Returns:
        Connection object to the initialized database
    """
    # First connect to the default postgres database
    default_conn = None
    final_conn = None
    
    try:
        # Connect to default postgres database
        default_conn = psycopg2.connect(
            host=host,
            port=port,
            dbname="postgres",  # Connect to default database
            user=user,
            password=password
        )
        default_conn.autocommit = True  # Required for CREATE DATABASE
        default_cursor = default_conn.cursor()
        
        # Check if our target database exists
        default_cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        db_exists = default_cursor.fetchone() is not None
        
        if not db_exists:
            print(f"Database '{dbname}' does not exist. Creating...")
            # Create database (must be done outside of transactions)
            default_cursor.execute(f"CREATE DATABASE {dbname}")
            print(f"Database '{dbname}' created successfully.")
        else:
            print(f"Database '{dbname}' already exists.")
            
        # Close default connection
        default_cursor.close()
        default_conn.close()
        
        # Connect to the target database
        final_conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        
        # Activate pgvector extension
        cursor = final_conn.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        vector_exists = cursor.fetchone() is not None
        
        if not vector_exists:
            print("Activating pgvector extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            final_conn.commit()
            print("pgvector extension activated successfully.")
        else:
            print("pgvector extension is already active.")
        
        # Check/create chunks table
        table_created = check_chunks_table(final_conn, vector_dimensions)
        if table_created:
            print("Chunks table created with vector search capabilities.")
        else:
            print("Chunks table already exists.")
            
        return final_conn
        
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        if default_conn and not default_conn.closed:
            default_conn.close()
        if final_conn and not final_conn.closed:
            final_conn.close()
        raise e

def insert_text_fragments(connection, text_fragments, keywords_lists, api_key, base_url_embeddings=None, model="text-embedding-3-small", batch_size=10):
    """
    Insert text fragments with their embeddings and keywords into the chunks table.
    
    Args:
        connection: PostgreSQL database connection
        text_fragments: List of text fragments to insert
        keywords_lists: List of keyword lists corresponding to each text fragment
        api_key: OpenAI API key for generating embeddings
        base_url_embeddings: Base URL for the OpenAI API (optional)
        model: The OpenAI embedding model to use
        batch_size: Number of fragments to process in each batch for better progress tracking
        
    Returns:
        Number of fragments successfully inserted
    """
    cursor = connection.cursor()
    inserted_count = 0
    
    try:
        # Initialize OpenAIEmbeddings once for all fragments
        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url_embeddings
        )
        
        # Process fragments in batches to provide progress updates
        batches = [text_fragments[i:i + batch_size] for i in range(0, len(text_fragments), batch_size)]
        keywords_batches = [keywords_lists[i:i + batch_size] for i in range(0, len(keywords_lists), batch_size)]
        
        for batch_idx, (fragments_batch, keywords_batch) in tqdm(enumerate(zip(batches, keywords_batches)), total=len(batches), desc="Processing embeddings and inserting fragments"):
            
            # Process each fragment in the current batch
            for i, (fragment, keywords) in enumerate(zip(fragments_batch, keywords_batch)):
                # Skip empty fragments
                if not fragment or fragment.strip() == "":
                    continue
                
                # Calculate embedding for the fragment using langchain
                embedding = embeddings.embed_documents([fragment])[0]
                
                # Prepare metadata with keywords
                metadata = {
                    "keywords": keywords,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Insert fragment with embedding and metadata
                cursor.execute("""
                    INSERT INTO chunks (content, content_vector, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                """, (
                    fragment,
                    embedding,
                    json.dumps(metadata)
                ))
                
                inserted_id = cursor.fetchone()[0]
                inserted_count += 1
                
                if (i + 1) % 5 == 0 or (i + 1) == len(fragments_batch):
                    # Commit after every few inserts to avoid large transactions
                    connection.commit()
            
            # Commit any remaining inserts in the batch
            connection.commit()
        
        print(f"Successfully inserted {inserted_count} fragments with embeddings and keywords.")
        return inserted_count
        
    except Exception as e:
        connection.rollback()
        print(f"Error inserting fragments: {e}")
        raise e
    finally:
        cursor.close()

def reciprocal_rank_fusion(results, top_k= 10, k_constant= 60.0):
    """
    Applies Reciprocal Rank Fusion to combine and rerank results from different retrieval methods.
    
    This implementation uses a modified RRF where the score is based on both the rank and the 
    original relevance score to better preserve the quality signals from each method.
    
    Args:
        results: List of dictionaries containing search results from hybrid_search
        top_k: Number of top results to return after fusion
        k_constant: Constant used in RRF formula (default: 60)
        
    Returns:
        List of reranked results limited to top_k items
    """
    # Count initial results by type
    vector_results = [r for r in results if r['search_type'] == 'vector']
    text_results = [r for r in results if r['search_type'] == 'text']
    
    print(f"Input results: {len(results)} total ({len(vector_results)} vector, {len(text_results)} text)")
    
    # Create a dictionary to store document IDs and their ranks/scores from different methods
    doc_scores = {}
    
    # Process vector results
    for rank, result in enumerate(sorted(vector_results, key=lambda x: x['score'], reverse=True)):
        doc_id = result['id']
        score = result['score']
        
        # Calculate RRF score component: 1/(k + rank)
        rrf_vector = 1.0 / (k_constant + rank + 1)  # +1 to make rank 1-based
        
        # Store with the original score too
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                'vector_rank': rank + 1,  
                'vector_score': score,
                'vector_rrf': rrf_vector,
                'document': result  # Keep the original document
            }
    
    # Process text results
    for rank, result in enumerate(sorted(text_results, key=lambda x: x['score'], reverse=True)):
        doc_id = result['id']
        score = result['score']
        
        # Calculate RRF score component
        rrf_text = 1.0 / (k_constant + rank + 1)
        
        # Update existing entry or create new one
        if doc_id in doc_scores:
            doc_scores[doc_id].update({
                'text_rank': rank + 1,
                'text_score': score,
                'text_rrf': rrf_text
            })
        else:
            doc_scores[doc_id] = {
                'text_rank': rank + 1,
                'text_score': score,
                'text_rrf': rrf_text,
                'document': result
            }
    
    # Calculate final RRF scores
    for doc_id, data in doc_scores.items():
        # Add RRF components with weight from original scores
        vector_component = data.get('vector_rrf', 0) * (data.get('vector_score', 0) / 100)
        text_component = data.get('text_rrf', 0) * (data.get('text_score', 0) / 100)
        
        # Final score is sum of components
        rrf_score = vector_component + text_component
        
        # For normalization, also track if document appeared in both methods
        methods_count = (1 if 'vector_rrf' in data else 0) + (1 if 'text_rrf' in data else 0)
        
        # Boost documents that appear in both methods
        if methods_count == 2:
            rrf_score *= 1.2  # 20% boost for documents found by both methods
        
        # Scale to 0-100 range for consistency with other scores
        data['rrf_score'] = rrf_score * 100
        data['methods_count'] = methods_count
    
    # Sort by RRF score and select top_k
    reranked_results = sorted(
        [
            {
                **data['document'],  # Original document data
                'score': data['rrf_score'],  # Update score with RRF score
                'original_score': data['document']['score'],  # Keep original score
                'methods_count': data['methods_count'],  # How many methods found this document
                'search_type': 'hybrid'  # Mark as hybrid result
            }
            for data in doc_scores.values()
        ],
        key=lambda x: x['score'],
        reverse=True
    )[:top_k]
    
    print(f"After fusion: {len(reranked_results)} results from {len(doc_scores)} unique documents")
    
    return reranked_results

# With no TOKEN streaming
def get_openai_response(prompt, api_key, base_url, model, temperature=0.3):
    """
    Get a response from OpenAI API for a given prompt.
    
    Args:
        prompt: Text prompt to send to the model
        api_key: OpenAI API key
        base_url: Base URL for the OpenAI API (optional)
        model: The model to use for completion
        temperature: Controls randomness (0-1), lower is more deterministic
        
    Returns:
        The text response from the model
    """
    try:
        # Configure client
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Call the API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear and concise answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        # Extract the message content
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message.content
            return message.strip()
        else:
            return "No response generated."
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise e

def answer_question_with_context(query, context_docs, api_key, base_url, model, prompt_template):
    """
    Generate an answer to a question using retrieved context documents.
    
    Args:
        query: The user's question
        context_docs: List of retrieved documents to use as context
        api_key: OpenAI API key
        base_url: Base URL for the OpenAI API (optional)
        model: The model to use for completion
        prompt_template: Optional custom prompt template with {query} and {context} placeholders
        
    Returns:
        The generated answer from the model
    """
    # Format context documents
    context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" 
                                    for i, doc in enumerate(context_docs)])
    
    # Format the prompt by replacing placeholders
    prompt = prompt_template.format(query=query, context=context)
    
    # Get response from OpenAI
    return get_openai_response(prompt, api_key, base_url, model)

# With TOKEN streaming
def get_openai_response_streaming(prompt, api_key, base_url, model, temperature=0.3):
    """
    Get a streaming response from OpenAI API for a given prompt.
    
    Args:
        prompt: Text prompt to send to the model
        api_key: OpenAI API key
        base_url: Base URL for the OpenAI API
        model: The model to use for completion
        temperature: Controls randomness (0-1), lower is more deterministic
        
    Returns:
        Generator that yields each token as it's generated
    """
    try:
        # Configure client
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Call the API with streaming enabled
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear and concise answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stream=True  # Enable streaming
        )
        
        # Return a generator that yields each token
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
            
    except Exception as e:
        print(f"Error calling OpenAI API with streaming: {e}")
        yield f"Error: {str(e)}"

def answer_question_with_context_streaming(query, context_docs, api_key, base_url, model, prompt_template):
    """
    Generate an answer to a question using retrieved context documents with streaming.
    
    Args:
        query: The user's question
        context_docs: List of retrieved documents to use as context
        api_key: OpenAI API key
        base_url: Base URL for the OpenAI API
        model: The model to use for completion
        prompt_template: Custom prompt template with {query} and {context} placeholders
        
    Returns:
        Generator that yields each token of the answer as it's generated
    """
    # Format context documents
    context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" 
                                    for i, doc in enumerate(context_docs)])
    
    # Format the prompt by replacing placeholders
    prompt = prompt_template.format(query=query, context=context)
    
    # Get streaming response from OpenAI
    return get_openai_response_streaming(prompt, api_key, base_url, model)

def process_text_into_chunks(fulltext):
    """
    Process a text string into logical chunks by:
    1. Joining lines where the next line doesn't start with an uppercase letter
    2. Splitting by double newlines
    3. Further splitting where a period is followed by uppercase letter
    
    Args:
        fulltext: The original text to process
        
    Returns:
        A list of text chunks in the original logical reading order
    """
    # Step 1: Join lines where the next line doesn't start with uppercase
    lines = fulltext.split('\n')
    processed_text = ""
    
    for i in range(len(lines) - 1):
        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()
        
        if current_line and next_line:
            # Check if next line starts with uppercase letter
            if next_line[0].isupper():
                processed_text += current_line + '\n'
            else:
                processed_text += current_line + ' '
        else:
            # Keep empty lines as they are
            processed_text += current_line + '\n'
    
    # Add the last line
    processed_text += lines[-1] if lines else ""
    
    # Step 2: Split by double newlines
    paragraphs = re.split(r'\n\s*\n', processed_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Step 3: Further split at periods followed by uppercase letters
    final_chunks = []
    
    for paragraph in paragraphs:
        # Find all positions where a period is followed by whitespace and then uppercase
        sentence_splits = re.finditer(r'\.(\s+)([A-Z])', paragraph)
        split_positions = [match.start() + 1 for match in sentence_splits]
        
        if not split_positions:
            # No splits needed
            final_chunks.append(paragraph)
            continue
        
        # Perform the splits
        start_pos = 0
        for pos in split_positions:
            chunk = paragraph[start_pos:pos].strip()
            if chunk:
                final_chunks.append(chunk)
            start_pos = pos
        
        # Add the last chunk
        last_chunk = paragraph[start_pos:].strip()
        if last_chunk:
            final_chunks.append(last_chunk)
    
    return final_chunks

