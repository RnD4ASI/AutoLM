import json
import time
import traceback
from src.logging import get_logger
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
from src.dbbuilder import GraphBuilder
# Import necessary modules from smolagents and other libraries
from src.generator import Generator, MetaGenerator
from src.utility import DataUtility, AIUtility
from src.evaluator import Evaluator, RetrievalEvaluator

logger = get_logger(__name__)

class QueryProcessor:
    """
    Handles query preprocessing methods like rewriting and decomposition.
    Responsible for transforming raw queries into more effective forms for retrieval.
    """
    
    def __init__(self, generator: Optional[Generator] = None):
        """
        Initialize the QueryProcessor.
        
        Args:
            generator: Generator instance for text generation and completions
        """
        logger.debug("QueryProcessor initialization started")
        try:
            start_time = time.time()
            self.generator = generator if generator else Generator()
            self.metagenerator = MetaGenerator(generator=self.generator)
            logger.debug(f"QueryProcessor initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"QueryProcessor initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def rephrase_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5) -> str:
        """
        Method 1: Rephrase the query to be more effective for retrieval.
        Uses metaprompt if available, otherwise falls back to template.
        
        Args:
            query: Original query string
            model: Model to use for generation
            temperature: Temperature for generation
            
        Returns:
            str: Rewritten query optimized for retrieval
        """
        logger.debug(f"Rephrasing query with model={model}, temperature={temperature}")
        logger.debug(f"Original query: '{query[:100]}{'...' if len(query) > 100 else ''}'")  # Log first 100 chars
        start_time = time.time()
        try:
            rephrased_query = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="manipulation",
                action="rephrase",  # Using rephrase as the action for query rewriting
                prompt_id=1,  # Using a default prompt ID
                task_prompt=query,
                model=model,
                temperature=temperature,
                return_full_response=False
            )
            logger.debug(f"Query successfully rephrased in {time.time() - start_time:.2f} seconds")
            logger.debug(f"Rephrased query: '{rephrased_query[:100]}{'...' if len(rephrased_query) > 100 else ''}'")  # Log first 100 chars
        except Exception as e:
            logger.error(f"Metaprompt query rephrasing failed, falling back to template: {e}")
            logger.debug(f"Rephrasing error details: {traceback.format_exc()}")
            rephrased_query = query  # Fall back to original query
            logger.debug("Using original query as fallback")
        return rephrased_query


    def decompose_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.7) -> List[Dict[str, Any]]:
        """
        Method 2: Decompose complex query into simpler sub-queries.
        Uses metaprompt if available, otherwise falls back to template.
        
        Args:
            query: Complex query string to decompose
            model: Model to use for generation
            temperature: Temperature for generation
            
        Returns:
            List[Dict[str, Any]]: List of sub-queries with their weights
        """
        try:
            decomposed_query = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="decompose",
                    prompt_id=2,  # Using a default prompt ID
                    task_prompt=query,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
        except Exception as e:
            logger.error(f"Query decomposition failed completely: {e}")
        return decomposed_query


    def hypothesize_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5) -> str:
        """
        Method 3: Apply HyDE to generate a hypothetical documentation as initial anchor for answering the task prompt.
        
        Args:
            query: Task prompt string to generate hypothetical query for
            model: Model to use for generation
            temperature: Temperature for generation
            
        Returns:
            str: Hypothetical query generated for the task prompt
        """
        try:
            hypothesized_query = self.generator.get_completion(
                    prompt_id=3,  # Using a default prompt ID
                    task_prompt=query,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
        except Exception as e:
            logger.error(f"Hypothetical query generation failed completely: {e}")
        return hypothesized_query
    

    def predict_query(self, query: str, response: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5):
        """
        Method 4: Generate a prediction of the next subquestion to solve, after considering the original task prompt, the prior subquestions, as well as the response to those prior subquestions.

        Args:
            query: Task prompt string of the original task
            response: Response to the subquestions of the original task and the corresponding responses
            model: Model to use for prediction
            temperature: Temperature for prediction
            
        Returns:
            str: Predicted query for the next subquestion
        """
        try:
            predicted_query = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="predict",
                    prompt_id=4,  # Using a default prompt ID
                    task_prompt=query,
                    response=response,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
        except Exception as e:
            logger.error(f"Predicted query generation failed completely: {e}")
        return predicted_query

class RerankProcessor:
    """
    Handles reranking of retrieval results using various methods.
    Responsible for improving the ordering of retrieved documents.
    """
    
    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "mps"):
        """
        Initialize the RerankProcessor.
        
        Args:
            reranker_model: HuggingFace model ID for reranking
        """
        logger.debug(f"Initializing RerankProcessor with model: {reranker_model}")
        start_time = time.time()
        
        # Initialize reranker
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        try:
            self.reranker = CrossEncoder(reranker_model, local_files_only=True, device = device)
            logger.info(f"Loaded reranker model: {reranker_model}")
            logger.debug(f"Reranker model information: device={device}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            logger.debug(f"Reranker load error details: {traceback.format_exc()}")
            self.reranker = None
        
        logger.debug(f"RerankProcessor initialization completed in {time.time() - start_time:.2f} seconds")
    
    def rerank_reciprocal_rank_fusion(self, results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to rerank results.
        Sam doesn't recommend this given it butched the unit of the measure.
        
        Args:
            results: List of retrieval results
            k: Constant in RRF formula (default: 60)
            
        Returns:
            List[Dict[str, Any]]: Reranked results
        """
        logger.debug(f"Starting reciprocal rank fusion reranking with k: {k}")
        start_time = time.time()
        
        # Sort results by score
        sorted_results = sorted(
            results,
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Calculate RRF scores
        for i, result in enumerate(sorted_results):
            rrf_score = 1 / (k + i + 1)
            result['score'] = rrf_score
        
        logger.debug(f"Reciprocal rank fusion reranking completed in {time.time() - start_time:.2f} seconds")
        return sorted_results
    
    def rerank_cross_encoder(self, query: str, results: List[Dict[str, Any]]):
        """
        Rerank results using a cross-encoder model.
        
        Args:
            query: Original query string
            results: List of retrieval results
            
        Returns:
            List[Dict[str, Any]]: Reranked results
        """
        logger.debug(f"Starting cross-encoder reranking for query: '{query[:50]}{'...' if len(query) > 50 else ''}' with {len(results)} results")
        start_time = time.time()
        
        if not self.reranker or not results:
            logger.debug(f"Skipping reranking: {'No reranker available' if not self.reranker else 'No results to rerank'}")
            return results
            
        try:
            # Create query-passage pairs for reranking
            logger.debug("Creating query-passage pairs for reranking")
            pairs = [(query, result['content']) for result in results]
            
            # Get reranking scores
            logger.debug(f"Getting cross-encoder scores for {len(pairs)} pairs")
            scores = self.reranker.predict(pairs)
            logger.debug(f"Received scores range: min={min(scores) if scores else 'N/A'}, max={max(scores) if scores else 'N/A'}")
            
            # Add scores to results
            for i, score in enumerate(scores):
                results[i]['cross_encoder_score'] = float(score)
                
            # Sort by score
            reranked_results = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)
            
            # Log the change in ranking
            if len(results) > 1:
                original_top_id = results[0].get('id', 'unknown')
                reranked_top_id = reranked_results[0].get('id', 'unknown')
                if original_top_id != reranked_top_id:
                    logger.debug(f"Reranking changed top result from id={original_top_id} to id={reranked_top_id}")
                else:
                    logger.debug("Top result remained the same after reranking")
            
            logger.debug(f"Cross-encoder reranking completed in {time.time() - start_time:.2f} seconds")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            logger.debug(f"Reranking error details: {traceback.format_exc()}")
            return results

class VectorDBRetrievalProcessor:
    """Implements retrieval methods for vector databases.
    
    This processor requires a vector database file to operate. It can be provided
    either through a VectorBuilder instance or directly as a file path.
    """
    
    def __init__(self, vector_builder=None, vector_file=None, generator=None):
        """Initialize VectorDBRetrievalProcessor.
        
        Args:
            generator: Generator instance for embeddings and completions
            embedding_model: Name of the embedding model to use for semantic search
            graph_builder: Optional GraphBuilder instance
        """
        self.generator = generator if generator else Generator()
        self.metagenerator = MetaGenerator(generator=self.generator)
        self.evaluator = Evaluator()
        self.aiutility = AIUtility()
        
        # Set embedding model - this can be overridden in search methods
        self.embedding_model = embedding_model
        self.embedding_model_instance = None  # Lazy-loaded when needed
        
        # Load NLP model for text processing
        try:
            if spacy.util.is_package("en_core_web_sm"):
                self.nlp = spacy.load("en_core_web_sm")
            else:
                logger.warning("Downloading spacy model 'en_core_web_sm'...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load spacy model: {e}")
        
        # Initialize symbolic search components
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2)
        )
        self.bm25 = None
        logger.debug("Retrieval Processor initialized")
        
        self.graph_query_processor = GraphQueryProcessor(graph_builder)
        
    def _load_embedding_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        Load embedding model on demand.
        
        Args:
            model_name: Name of the model to load, defaults to self.embedding_model
            
        Returns:
            SentenceTransformer: The loaded embedding model
        """
        model_to_load = model_name if model_name else self.embedding_model
        
        try:
            return SentenceTransformer(model_to_load)
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_to_load}: {e}")
            # Fall back to generator's embedding functionality
            return None
    
    def semantic_search(self, query: Union[str, np.ndarray], corpus: List[Dict[str, Any]], 
                     top_k: int = None, top_p: float = None, 
                     embedding_model: str = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:  
        """
        From first principles: Retrieve documents using semantic vector similarity.
        
        Args:
            query: Query string or embedding vector
            corpus: List of documents to search in, each with content and optional embedding
            top_k: Number of top results to return (optional)
            top_p: Probability threshold for results (optional)
            embedding_model: Name of embedding model to use (optional, uses default if None)
            
        Returns:
            Tuple containing:
                - List of corpus items (filtered if top_k or top_p is specified)
                - List of metadata dictionaries with scores
        """
        if not corpus:
            logger.warning("Corpus is empty")
            return [], []
            
        try:
            # Get query embedding if string was provided
            query_embedding = None
            if isinstance(query, str):
                if embedding_model and embedding_model != self.embedding_model:
                    model = self._load_embedding_model(embedding_model)
                    if model:
                        query_embedding = model.encode(query)
                    else:
                        # Fall back to generator's embedding functionality
                        query_embedding = self.generator.get_embedding(query)
                else:
                    # Use default embedding method
                    query_embedding = self.generator.get_embedding(query)
            else:
                # Query is already an embedding vector
                query_embedding = query
            
            # Calculate similarities
            similarities = []
            result_corpus = []
            metadata_results = []
            
            for i, doc in enumerate(corpus):
                # Get document embedding
                doc_embedding = None
                if 'embedding' in doc:
                    doc_embedding = np.array(doc['embedding'])
                else:
                    # Generate embedding if not present
                    if embedding_model and embedding_model != self.embedding_model:
                        model = self._load_embedding_model(embedding_model)
                        if model:
                            doc_embedding = model.encode(doc['content'])
                        else:
                            doc_embedding = self.generator.get_embedding(doc['content'])
                    else:
                        doc_embedding = self.generator.get_embedding(doc['content'])
                
                # Calculate similarity score
                sim_score = 1 - cosine(query_embedding, doc_embedding)
                
                similarities.append(sim_score)
                metadata_results.append({
                    'id': doc.get('id', str(i)),
                    'metadata': doc.get('metadata', {}),
                    'score': float(sim_score),
                    'source': 'semantic'
                })
                result_corpus.append(doc)
            
            # Apply filtering if specified
            if top_k or top_p:
                # Convert to numpy array if not already
                if not isinstance(similarities, np.ndarray):
                    similarities = np.array(similarities)
                
                # Get sorted indices (descending order)
                sorted_indices = np.argsort(similarities)[::-1]
                
                # Apply top-p filter if specified
                if top_p is not None:
                    filtered_indices = sorted_indices[similarities[sorted_indices] >= top_p]
                else:
                    filtered_indices = sorted_indices
                
                # Apply top-k filter if specified
                if top_k is not None and len(filtered_indices) > top_k:
                    filtered_indices = filtered_indices[:top_k]
                
                # Convert indices to list
                filtered_indices = filtered_indices.tolist()
                
                # Filter results
                result_corpus = [result_corpus[i] for i in filtered_indices]
                metadata_results = [metadata_results[i] for i in filtered_indices]
            
            return result_corpus, metadata_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return [], []
    
    def symbolic_search(
        self, query: str, 
        corpus: List[Dict[str, Any]], 
        method: str = "tfidf", 
        top_k: int = None, 
        top_p: float = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:  
        """
        From first principles: Retrieve documents using symbolic scoring (TF-IDF or BM25).
        
        Args:
            query: Query string
            corpus: List of documents to search in
            method: Scoring method ('tfidf' or 'bm25')
            top_k: Number of top results to return (optional)
            top_p: Probability threshold for results (optional)
            
        Returns:
            Tuple containing:
                - List of corpus items (filtered if top_k or top_p is specified)
                - List of metadata dictionaries with scores
        """
        if not corpus:
            logger.warning("Corpus is empty")
            return [], []
        
        try:
            # Extract content from corpus for vectorization
            corpus_content = [doc['content'] for doc in corpus]
            
            if method == "tfidf":
                # Fit vectorizer if not already fitted
                if not self.tfidf_vectorizer.get_feature_names_out:
                    self.tfidf_vectorizer.fit(corpus_content)
                    
                # Transform query and documents
                query_vec = self.tfidf_vectorizer.transform([query])
                doc_vectors = self.tfidf_vectorizer.transform(corpus_content)
                
                # Calculate cosine similarities
                similarities = (query_vec * doc_vectors.T).toarray()[0]
                
            else:  # BM25
                # Initialize BM25 if not already done
                tokenized_corpus = [self.nlp(doc).text.split() for doc in corpus_content]
                bm25 = BM25Okapi(tokenized_corpus)
                
                # Tokenize query
                tokenized_query = self.nlp(query).text.split()
                
                # Get BM25 scores
                similarities = bm25.get_scores(tokenized_query)
                
                # Normalize BM25 scores to 0-1 range for consistency
                max_score = max(similarities) if similarities.any() else 1
                similarities = similarities / max_score if max_score > 0 else similarities
            
            # Create results list
            metadata_results = []
            for i, (doc, score) in enumerate(zip(corpus, similarities)):
                metadata_results.append({
                    'id': doc.get('id', str(i)),
                    'metadata': doc.get('metadata', {}),
                    'score': float(score),
                    'source': method
                })
            
            # Apply filtering if specified
            result_corpus = corpus
            if top_k or top_p:
                # Convert to numpy array if not already
                if not isinstance(similarities, np.ndarray):
                    similarities = np.array(similarities)
                
                # Get sorted indices (descending order)
                sorted_indices = np.argsort(similarities)[::-1]
                
                # Apply top-p filter if specified
                if top_p is not None:
                    filtered_indices = sorted_indices[similarities[sorted_indices] >= top_p]
                else:
                    filtered_indices = sorted_indices
                
                # Apply top-k filter if specified
                if top_k is not None and len(filtered_indices) > top_k:
                    filtered_indices = filtered_indices[:top_k]
                
                # Convert indices to list
                filtered_indices = filtered_indices.tolist()
                
                # Filter results
                result_corpus = [corpus[i] for i in filtered_indices]
                metadata_results = [metadata_results[i] for i in filtered_indices]
            
            return result_corpus, metadata_results
            
        except Exception as e:
            logger.error(f"Symbolic search failed: {e}")
            return [], []
    
    def graph_based_search(self, query: str, 
                          search_type: str = 'semantic',
                          **kwargs) -> List[Dict[str, Any]]:
        """Perform graph-based search using different strategies.
        
        Args:
            query: Search query
            search_type: One of 'semantic', 'concept', 'temporal', 'cross_layer'
            **kwargs: Additional arguments for specific search types
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        if not self.graph_query_processor:
            raise ValueError("Graph query processor not initialized")
            
        if search_type == 'semantic':
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            return self.graph_query_processor.semantic_cluster_search(
                query_embedding, **kwargs)
                
        elif search_type == 'concept':
            return self.graph_query_processor.concept_hierarchy_search(
                query, **kwargs)
                
        elif search_type == 'temporal':
            return self.graph_query_processor.temporal_search(**kwargs)
                
        elif search_type == 'cross_layer':
            return self.graph_query_processor.cross_layer_search(
                query, **kwargs)
                
        else:
            raise ValueError(f"Unknown search type: {search_type}")
  
class GraphDBRetrievalProcessor:
    """
    Implements specialized query methods for graph and hypergraph structures.
    Supports querying across different graph representations and layers.
    
    This processor requires a graph database file to operate. It can be provided
    either through a GraphBuilder instance or directly as a file path.
    """
    
    def __init__(self, graph_builder=None, graph_file=None, graph_type='standard'):
        """Initialize GraphDBRetrievalProcessor.
        
        Args:
            graph_builder: Optional GraphBuilder instance
            graph_file: Optional path to a graph database file (.pkl)
            graph_type: Type of graph to load ('standard', 'hypergraph', 'semantic', 'multilayer')
        """
        logger.debug("GraphDBRetrievalProcessor initialization started")
        try:
            start_time = time.time()
            self.nx = __import__('networkx')
            
            # Handle graph builder or file path
            if graph_builder:
                self.graph_builder = graph_builder
                logger.debug(f"Using provided graph_builder of type {type(graph_builder).__name__}")
            elif graph_file:
                # Import GraphBuilder here to avoid circular imports
                
                self.graph_builder = GraphBuilder()
                self.graph_builder.load_graph_db(graph_file, graph_type=graph_type)
                logger.debug(f"Loaded graph from file: {graph_file}")
            else:
                self.graph_builder = None
                logger.debug("No graph_builder or graph_file provided, some functionality may be limited")
                
            logger.debug(f"GraphDBRetrievalProcessor initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"GraphDBRetrievalProcessor initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise

    def semantic_cluster_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search through semantic clusters in the hypergraph.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Ranked list of relevant sections
        """
        logger.debug(f"Starting semantic_cluster_search with top_k={top_k}")
        start_time = time.time()
        
        try:
            if not hasattr(self.graph_builder, 'semantic_hypergraph'):
                logger.error("Semantic hypergraph not built")
                raise ValueError("Semantic hypergraph not built")
                
            H = self.graph_builder.semantic_hypergraph
            results = []
            
            # Find semantic cluster nodes
            logger.debug("Finding semantic cluster nodes in hypergraph")
            semantic_clusters = [n for n, d in H.nodes(data=True) 
                               if d.get('type') == 'hyperedge' and 
                               d.get('edge_type') == 'semantic_cluster']
            logger.debug(f"Found {len(semantic_clusters)} semantic cluster nodes")
            
            for i, cluster in enumerate(semantic_clusters):
                # Get sections in this cluster
                sections = [n for n in H.neighbors(cluster) if H.nodes[n].get('type') == 'section']
                logger.debug(f"Cluster {i+1}/{len(semantic_clusters)} has {len(sections)} sections")
                
                # Calculate average similarity to query
                similarities = []
                for section in sections:
                    if 'embedding' in H.nodes[section]:
                        sim = 1 - cosine(query_embedding, H.nodes[section]['embedding'])
                        similarities.append((section, sim))
            
                if similarities:
                    # Add top section from each relevant cluster
                    best_section, best_sim = max(similarities, key=lambda x: x[1])
                    
                    # Add to results
                    results.append({
                        'id': best_section,
                        'score': best_sim,
                        'content': H.nodes[best_section].get('content', ''),
                        'metadata': {
                            'source': 'semantic_cluster',
                            'cluster_id': cluster
                        }
                    })
                    
            # Sort results by similarity score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Truncate to top_k
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
            logger.debug(f"Semantic cluster search completed in {time.time() - start_time:.2f} seconds, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic cluster search failed: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            return []

    def concept_hierarchy_search(self, query: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Search through concept hierarchies in the semantic hypergraph.
        
        Args:
            query: Search query
            max_depth: Maximum depth to traverse in concept hierarchy
            
        Returns:
            List[Dict[str, Any]]: Related concepts and sections
        """
        if not hasattr(self.graph_builder, 'semantic_hypergraph'):
            raise ValueError("Semantic hypergraph not built")
            
        H = self.graph_builder.semantic_hypergraph
        results = []
        
        # Find concept hierarchy edges
        concept_edges = [n for n, d in H.nodes(data=True)
                        if d.get('type') == 'hyperedge' and
                        d.get('edge_type') == 'concept_hierarchy']
        
        def traverse_concepts(edge, depth=0):
            if depth >= max_depth:
                return
                
            edge_data = H.nodes[edge]
            parent = edge_data.get('parent_concept', '')
            children = edge_data.get('child_concepts', [])
            
            # Check if query matches any concepts
            if (query.lower() in parent.lower() or 
                any(query.lower() in child.lower() for child in children)):
                
                # Get sections connected to this concept
                sections = [n for n in H.neighbors(edge) if H.nodes[n].get('type') == 'section']
                results.append({
                    'parent_concept': parent,
                    'child_concepts': children,
                    'sections': sections,
                    'depth': depth
                })
                
                # Recursively check related concepts
                for child in children:
                    child_edges = [n for n, d in H.nodes(data=True)
                                 if d.get('type') == 'hyperedge' and
                                 d.get('edge_type') == 'concept_hierarchy' and
                                 d.get('parent_concept') == child]
                    for child_edge in child_edges:
                        traverse_concepts(child_edge, depth + 1)
        
        for edge in concept_edges:
            traverse_concepts(edge)
            
        return results

    def temporal_search(self, start_date: str, end_date: str = None) -> List[Dict[str, Any]]:
        """Search for content within a time period in the multilayer hypergraph.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: Optional end date string (YYYY-MM-DD)
            
        Returns:
            List[Dict[str, Any]]: Time-relevant sections
        """
        if not hasattr(self.graph_builder, 'multilayer_hypergraph'):
            raise ValueError("Multilayer hypergraph not built")
            
        layers = self.graph_builder.multilayer_hypergraph
        metadata_layer = layers.get('metadata')
        if not metadata_layer:
            raise ValueError("Metadata layer not found")
            
        results = []
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else None
        
        # Find temporal nodes in metadata layer
        temporal_nodes = [n for n, d in metadata_layer.nodes(data=True)
                         if d.get('type') == 'metadata' and d.get('field') == 'date']
        
        for node in temporal_nodes:
            node_data = metadata_layer.nodes[node]
            node_date = pd.to_datetime(node_data.get('value'))
            
            if start <= node_date and (not end or node_date <= end):
                # Get sections connected to this date
                sections = [n for n in metadata_layer.neighbors(node)]
                results.append({
                    'date': node_date,
                    'sections': sections
                })
        
        return sorted(results, key=lambda x: x['date'])

    def cross_layer_search(self, query: str, layer_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Search across multiple layers of the multilayer hypergraph.
        
        Args:
            query: Search query
            layer_weights: Optional weights for each layer (default: equal weights)
            
        Returns:
            List[Dict[str, Any]]: Cross-layer search results
        """
        if not hasattr(self.graph_builder, 'multilayer_hypergraph'):
            raise ValueError("Multilayer hypergraph not built")
            
        layers = self.graph_builder.multilayer_hypergraph
        if not layer_weights:
            layer_weights = {name: 1.0 for name in layers.keys()}
            
        results = {}
        
        for layer_name, layer in layers.items():
            weight = layer_weights.get(layer_name, 1.0)
            
            if layer_name == 'content':
                # Use semantic similarity for content layer
                matches = self._content_layer_search(layer, query)
            elif layer_name == 'structure':
                # Use hierarchy traversal for structure layer
                matches = self._structure_layer_search(layer, query)
            elif layer_name == 'reference':
                # Use reference graph traversal
                matches = self._reference_layer_search(layer, query)
            else:
                # Use basic attribute matching for other layers
                matches = self._attribute_layer_search(layer, query)
                
            # Combine scores across layers
            for section_id, score in matches:
                if section_id not in results:
                    results[section_id] = 0
                results[section_id] += score * weight
        
        # Return normalized results
        return [{'reference': k, 'score': v} for k, v in 
                sorted(results.items(), key=lambda x: x[1], reverse=True)]

    def _content_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within content layer using semantic similarity."""
        results = []
        for node, data in layer.nodes(data=True):
            if data.get('type') == 'section' and 'embedding' in data:
                sim = 1 - cosine(query_embedding, data['embedding'])
                results.append((node, sim))
        return results

    def _structure_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within structure layer using hierarchy."""
        results = []
        for node, data in layer.nodes(data=True):
            if data.get('type') == 'section':
                # Score based on hierarchy level match
                hierarchy = data.get('hierarchy', '').lower()
                if query.lower() in hierarchy:
                    level = data.get('level', 0)
                    score = 1.0 / (level + 1)  # Higher score for top levels
                    results.append((node, score))
        return results

    def _reference_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within reference layer using graph traversal."""
        results = []
        # Find nodes matching query
        matches = [n for n in layer.nodes() if query.lower() in n.lower()]
        
        for match in matches:
            # Use PageRank to score connected nodes
            personalization = {match: 1.0}
            scores = self.nx.pagerank(layer, personalization=personalization)
            results.extend(scores.items())
        return results

    def _attribute_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within attribute-based layers (metadata, access)."""
        results = []
        for node, data in layer.nodes(data=True):
            if data.get('type') == 'section':
                # Score based on attribute matches
                matches = sum(1 for v in data.values() 
                            if isinstance(v, str) and query.lower() in v.lower())
                if matches:
                    results.append((node, matches))
        return results
        
    def path_based_search(self, start_node: str, end_node: str, 
                          max_path_length: int = 4) -> List[Dict[str, Any]]:
        """Find all paths between two nodes in the standard graph.
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_path_length: Maximum path length to consider
            
        Returns:
            List[Dict[str, Any]]: List of paths with metadata
        """
        if not hasattr(self.graph_builder, 'graph'):
            raise ValueError("Standard graph not built")
            
        G = self.graph_builder.graph
        results = []
        
        # Find all simple paths within length limit
        try:
            all_paths = list(self.nx.all_simple_paths(
                G, source=start_node, target=end_node, cutoff=max_path_length
            ))
            
            for i, path in enumerate(all_paths):
                path_edges = []
                path_weight = 0
                
                # Extract edge metadata
                for j in range(len(path) - 1):
                    src, dst = path[j], path[j+1]
                    edge_data = G.get_edge_data(src, dst)
                    
                    # Handle case of multiple edges between nodes
                    if isinstance(edge_data, dict) and len(edge_data) > 0:
                        # If multiple edges exist, choose the first one
                        if 0 in edge_data:
                            edge_attr = edge_data[0]
                        else:
                            # Get the first edge key and its data
                            first_key = list(edge_data.keys())[0]
                            edge_attr = edge_data[first_key]
                            
                        edge_type = edge_attr.get('type', 'unknown')
                        weight = edge_attr.get('weight', 1.0)
                        
                        path_edges.append({
                            'source': src,
                            'target': dst,
                            'type': edge_type,
                            'weight': weight
                        })
                        
                        path_weight += weight
                
                # Calculate path score (inversely proportional to length)
                path_score = path_weight / len(path)
                
                results.append({
                    'path': path,
                    'path_edges': path_edges,
                    'length': len(path) - 1,
                    'score': path_score
                })
            
            # Sort by path score
            return sorted(results, key=lambda x: x['score'], reverse=True)
            
        except (self.nx.NetworkXNoPath, self.nx.NodeNotFound) as e:
            logger.warning(f"No path found: {e}")
            return []
            
    def connectivity_search(self, nodes: List[str], 
                           connectivity_metric: str = 'shortest_path') -> Dict[str, Any]:
        """Analyze connectivity between a set of nodes.
        
        Args:
            nodes: List of node IDs to analyze
            connectivity_metric: One of 'shortest_path', 'clustering', 'centrality'
            
        Returns:
            Dict[str, Any]: Connectivity analysis results
        """
        if not hasattr(self.graph_builder, 'graph'):
            raise ValueError("Standard graph not built")
            
        G = self.graph_builder.graph
        valid_nodes = [n for n in nodes if n in G]
        
        if len(valid_nodes) < 2:
            return {'error': 'Need at least two valid nodes for connectivity analysis'}
            
        results = {
            'nodes': valid_nodes,
            'valid_node_count': len(valid_nodes),
            'analysis_type': connectivity_metric
        }
        
        if connectivity_metric == 'shortest_path':
            # Find shortest paths between all pairs
            paths = {}
            for i in range(len(valid_nodes)):
                for j in range(i+1, len(valid_nodes)):
                    src, dst = valid_nodes[i], valid_nodes[j]
                    try:
                        path = self.nx.shortest_path(G, source=src, target=dst)
                        length = len(path) - 1
                        paths[f"{src}_to_{dst}"] = {
                            'path': path,
                            'length': length
                        }
                    except (self.nx.NetworkXNoPath, self.nx.NodeNotFound):
                        paths[f"{src}_to_{dst}"] = {'path': [], 'length': float('inf')}
            
            # Calculate average path length
            finite_lengths = [p['length'] for p in paths.values() if p['length'] < float('inf')]
            avg_length = sum(finite_lengths) / len(finite_lengths) if finite_lengths else float('inf')
            
            results['paths'] = paths
            results['average_path_length'] = avg_length
            results['connectivity_score'] = 1.0 / avg_length if avg_length > 0 else 0
            
        elif connectivity_metric == 'clustering':
            # Extract subgraph
            subgraph = G.subgraph(valid_nodes)
            
            # Get clustering coefficient
            clustering = self.nx.clustering(subgraph)
            avg_clustering = sum(clustering.values()) / len(clustering) if clustering else 0
            
            results['node_clustering'] = clustering
            results['average_clustering'] = avg_clustering
            results['connectivity_score'] = avg_clustering
            
        elif connectivity_metric == 'centrality':
            # Extract subgraph
            subgraph = G.subgraph(valid_nodes)
            
            # Calculate various centrality measures
            degree_centrality = self.nx.degree_centrality(subgraph)
            betweenness_centrality = self.nx.betweenness_centrality(subgraph)
            
            results['degree_centrality'] = degree_centrality
            results['betweenness_centrality'] = betweenness_centrality
            results['connectivity_score'] = sum(degree_centrality.values()) / len(degree_centrality)
            
        else:
            results['error'] = f"Unknown connectivity metric: {connectivity_metric}"
            
        return results
        
    def community_detection(self, algorithm: str = 'louvain', 
                           resolution: float = 1.0) -> Dict[str, Any]:
        """Detect communities in the graph using various algorithms.
        
        Args:
            algorithm: One of 'louvain', 'label_propagation', 'greedy_modularity'
            resolution: Resolution parameter for community detection (for louvain)
            
        Returns:
            Dict[str, Any]: Community detection results
        """
        if not hasattr(self.graph_builder, 'graph'):
            raise ValueError("Standard graph not built")
            
        G = self.graph_builder.graph
        
        # Convert MultiDiGraph to simple undirected graph for community detection
        simple_G = self.nx.Graph()
        for u, v, data in G.edges(data=True):
            # If edge already exists, we don't add it again or increment
            if not simple_G.has_edge(u, v):
                simple_G.add_edge(u, v, weight=data.get('weight', 1.0))
                
        results = {
            'algorithm': algorithm,
            'node_count': simple_G.number_of_nodes()
        }
        
        try:
            communities = {}
            
            if algorithm == 'louvain':
                # Import community detection algorithm
                from community import best_partition
                
                partition = best_partition(simple_G, resolution=resolution)
                # Reformat partition to communities
                communities_dict = {}
                for node, community_id in partition.items():
                    if community_id not in communities_dict:
                        communities_dict[community_id] = []
                    communities_dict[community_id].append(node)
                
                communities = list(communities_dict.values())
                
            elif algorithm == 'label_propagation':
                communities = list(self.nx.algorithms.community.label_propagation_communities(simple_G))
                
            elif algorithm == 'greedy_modularity':
                communities = list(self.nx.algorithms.community.greedy_modularity_communities(simple_G))
                
            else:
                return {'error': f"Unknown community detection algorithm: {algorithm}"}
            
            # Calculate modularity
            modularity = self.nx.algorithms.community.modularity(simple_G, communities)
            
            results['communities'] = [list(c) for c in communities]
            results['community_count'] = len(communities)
            results['modularity'] = modularity
            
            # Calculate additional statistics
            community_sizes = [len(c) for c in communities]
            results['avg_community_size'] = sum(community_sizes) / len(community_sizes)
            results['max_community_size'] = max(community_sizes)
            results['min_community_size'] = min(community_sizes)
            
            return results
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {'error': f"Community detection failed: {str(e)}"}
            
    def hypergraph_query(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Query hypergraph structures with various specialized algorithms.
        
        Args:
            query_type: One of 'entity_influence', 'knowledge_gap', 'concept_diffusion'
            **kwargs: Additional arguments for specific query types
            
        Returns:
            List[Dict[str, Any]]: Hypergraph query results
        """
        if not hasattr(self.graph_builder, 'hypergraph'):
            raise ValueError("Hypergraph not built")
            
        H = self.graph_builder.hypergraph
        results = []
        
        if query_type == 'entity_influence':
            # Calculate influence of entities by hyperedge membership analysis
            min_hyperedges = kwargs.get('min_hyperedges', 3)
            
            # Count hyperedge membership for each section node
            membership_count = {}
            for node in H.nodes():
                if H.nodes[node].get('type') == 'section':
                    # Get all hyperedges this node belongs to
                    hyperedges = [n for n in H.neighbors(node) 
                                if H.nodes[n].get('type') == 'hyperedge']
                    membership_count[node] = len(hyperedges)
            
            # Filter by minimum hyperedge count
            influential_nodes = {node: count for node, count in membership_count.items() 
                               if count >= min_hyperedges}
            
            # Calculate influence score based on hyperedge diversity
            node_scores = {}
            for node, count in influential_nodes.items():
                hyperedges = [n for n in H.neighbors(node) if H.nodes[n].get('type') == 'hyperedge']
                
                # Measure diversity by counting different edge types
                edge_types = set(H.nodes[he].get('edge_type', 'unknown') for he in hyperedges)
                
                # Influence = membership count * edge type diversity
                node_scores[node] = count * len(edge_types)
            
            # Return sorted by influence score
            for node, score in sorted(node_scores.items(), key=lambda x: x[1], reverse=True):
                results.append({
                    'node': node,
                    'influence_score': score,
                    'hyperedge_count': influential_nodes[node]
                })
                
        elif query_type == 'knowledge_gap':
            # Identify potential gaps in the knowledge base
            min_cluster_size = kwargs.get('min_cluster_size', 3)
            max_connections = kwargs.get('max_connections', 2)
            
            # Find document clusters (hyperedges with document_group type)
            doc_clusters = [n for n, d in H.nodes(data=True)
                          if d.get('type') == 'hyperedge' and 
                          d.get('edge_type') == 'document_group']
            
            for cluster in doc_clusters:
                # Get sections in this document cluster
                sections = [n for n in H.neighbors(cluster)]
                
                if len(sections) >= min_cluster_size:
                    # Look for sections with few connections to other document clusters
                    for section in sections:
                        # Get all hyperedges this section belongs to
                        section_hyperedges = [n for n in H.neighbors(section)
                                           if H.nodes[n].get('type') == 'hyperedge']
                        
                        # Count connections to other document clusters
                        other_doc_connections = sum(1 for he in section_hyperedges
                                                if H.nodes[he].get('edge_type') == 'document_group'
                                                and he != cluster)
                        
                        if other_doc_connections <= max_connections:
                            results.append({
                                'section': section,
                                'document_cluster': cluster,
                                'cluster_size': len(sections),
                                'other_connections': other_doc_connections,
                                'gap_score': 1.0 / (other_connections + 1)
                            })
            
            # Sort by gap score
            results = sorted(results, key=lambda x: x['gap_score'], reverse=True)
                
        elif query_type == 'concept_diffusion':
            # Track how concepts spread through the hypergraph
            concept_name = kwargs.get('concept', '')
            max_distance = kwargs.get('max_distance', 3)
            
            if not concept_name:
                return [{'error': 'Concept name required for concept diffusion query'}]
                
            # Find hyperedges related to this concept
            concept_edges = [n for n, d in H.nodes(data=True)
                          if d.get('type') == 'hyperedge' and 
                          ((d.get('edge_type') == 'key_concept' and 
                           d.get('concept', '').lower() == concept_name.lower()) or
                           (d.get('edge_type') == 'concept_hierarchy' and 
                           (d.get('parent_concept', '').lower() == concept_name.lower() or
                            concept_name.lower() in [c.lower() for c in d.get('child_concepts', [])])))]
            
            if not concept_edges:
                return [{'error': f"Concept '{concept_name}' not found in the hypergraph"}]
                
            # Track diffusion from these concept edges
            diffusion_map = {}
            visited = set()
            
            def traverse_diffusion(node, distance=0):
                if distance > max_distance or node in visited:
                    return
                    
                visited.add(node)
                
                # Store node in diffusion map
                if distance not in diffusion_map:
                    diffusion_map[distance] = []
                    
                diffusion_map[distance].append(node)
                
                # Traverse neighbors
                for neighbor in H.neighbors(node):
                    traverse_diffusion(neighbor, distance + 1)
            
            # Start traversal from each concept edge
            for edge in concept_edges:
                traverse_diffusion(edge)
            
            # Format results
            for distance, nodes in sorted(diffusion_map.items()):
                hyperedges = [n for n in nodes if H.nodes[n].get('type') == 'hyperedge']
                sections = [n for n in nodes if H.nodes[n].get('type') == 'section']
                
                results.append({
                    'distance': distance,
                    'hyperedge_count': len(hyperedges),
                    'section_count': len(sections),
                    'hyperedges': hyperedges[:10],  # Limit to avoid huge results
                    'sections': sections[:10]       # Limit to avoid huge results
                })
                
        elif query_type == 'structural_patterns':
            # Detect recurring structural patterns in the hypergraph
            min_pattern_size = kwargs.get('min_pattern_size', 3)
            max_patterns = kwargs.get('max_patterns', 10)
            
            # Find patterns of interconnected hyperedges
            patterns = []
            
            # Group hyperedges by type
            hyperedge_types = {}
            for node, data in H.nodes(data=True):
                if data.get('type') == 'hyperedge':
                    edge_type = data.get('edge_type', 'unknown')
                    if edge_type not in hyperedge_types:
                        hyperedge_types[edge_type] = []
                    hyperedge_types[edge_type].append(node)
            
            # Compare patterns within each edge type
            for edge_type, edges in hyperedge_types.items():
                for i in range(len(edges)):
                    edge1 = edges[i]
                    neighbors1 = set(H.neighbors(edge1))
                    
                    if len(neighbors1) < min_pattern_size:
                        continue
                        
                    # Find similar hyperedges (with similar section connections)
                    for j in range(i+1, len(edges)):
                        edge2 = edges[j]
                        neighbors2 = set(H.neighbors(edge2))
                        
                        if len(neighbors2) < min_pattern_size:
                            continue
                            
                        # Calculate Jaccard similarity
                        intersection = len(neighbors1.intersection(neighbors2))
                        union = len(neighbors1.union(neighbors2))
                        
                        if union > 0:
                            similarity = intersection / union
                            
                            if similarity > 0.5:  # Significant overlap
                                common_nodes = neighbors1.intersection(neighbors2)
                                
                                if len(common_nodes) >= min_pattern_size:
                                    patterns.append({
                                        'edge_type': edge_type,
                                        'hyperedges': [edge1, edge2],
                                        'common_nodes': list(common_nodes),
                                        'similarity': similarity,
                                        'pattern_size': len(common_nodes)
                                    })
            
            # Sort by pattern size and similarity
            patterns.sort(key=lambda x: (x['pattern_size'], x['similarity']), reverse=True)
            results = patterns[:max_patterns]  # Limit number of patterns
            
        else:
            results = [{'error': f"Unknown hypergraph query type: {query_type}"}]
            
        return results
        
    def multilayer_hypergraph_fusion(self, query: str, 
                                    layer_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Perform fusion search across all layers in multilayer hypergraph.
        
        This advanced search combines results from different hypergraph layers,
        using an ensemble approach with customizable weights.
        
        Args:
            query: Search query
            layer_weights: Optional weights for each layer (default: equal weights)
            
        Returns:
            List[Dict[str, Any]]: Fusion search results
        """
        if not hasattr(self.graph_builder, 'multilayer_hypergraph'):
            raise ValueError("Multilayer hypergraph not built")
            
        layers = self.graph_builder.multilayer_hypergraph
        
        # Default layer weights (content prioritized)
        if layer_weights is None:
            layer_weights = {
                'content': 1.0,
                'structure': 0.7,
                'reference': 0.6,
                'metadata': 0.4,
                'access': 0.3
            }
        
        # Unified results dictionary (node_id -> score mapping)
        fusion_scores = {}
        layer_results = {}
        
        # Split query into semantic and keyword components
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        keywords = [word.lower() for word in query.split() if word.lower() not in stop_words]
        
        # Process each layer
        for layer_name, layer in layers.items():
            if layer_name not in layer_weights:
                continue
                
            # Layer-specific search strategy
            if layer_name == 'content':
                # Semantic search in content layer
                section_scores = {}
                
                # For each section node with embedding
                for node, data in layer.nodes(data=True):
                    if 'embedding' in data:
                        # Calculate similarity to query
                        text = data.get('text', '')
                        # Direct keyword matching (basic TF-IDF like scoring)
                        keyword_matches = 0
                        for keyword in keywords:
                            if keyword in text.lower():
                                keyword_matches += 1
                                
                        # Normalize by number of keywords
                        score = keyword_matches / len(keywords) if keywords else 0
                        section_scores[node] = score
                        
                layer_results[layer_name] = section_scores
                
            elif layer_name == 'structure':
                # Hierarchical search
                section_scores = {}
                
                # Search through hierarchy nodes
                for node, data in layer.nodes(data=True):
                    hierarchy = data.get('hierarchy', '')
                    level = data.get('level', 0)
                    
                    # Match query against hierarchy
                    if any(keyword in hierarchy.lower() for keyword in keywords):
                        # Score inversely proportional to level depth
                        score = 1.0 / (level + 1)
                        section_scores[node] = score
                        
                layer_results[layer_name] = section_scores
                
            elif layer_name == 'reference':
                # Reference graph analysis
                section_scores = {}
                
                # Find initial matches
                matched_nodes = [node for node in layer.nodes() 
                               if any(keyword in str(node).lower() for keyword in keywords)]
                
                # Use PageRank with personalization to extend from matches
                if matched_nodes:
                    personalization = {node: 1.0 for node in matched_nodes}
                    try:
                        scores = self.nx.pagerank(layer, personalization=personalization)
                        section_scores.update(scores)
                    except:
                        # Fallback if PageRank fails
                        for node in matched_nodes:
                            section_scores[node] = 1.0
                            # Add immediate neighbors with reduced scores
                            for neighbor in layer.neighbors(node):
                                section_scores[neighbor] = 0.5
                                
                layer_results[layer_name] = section_scores
                
            elif layer_name in ('metadata', 'access'):
                # Attribute matching for metadata/access layers
                section_scores = {}
                
                # Search through attributes
                for node, data in layer.nodes(data=True):
                    # Check all string attributes for keyword matches
                    matches = sum(1 for k, v in data.items() 
                                if isinstance(v, str) and 
                                any(keyword in v.lower() for keyword in keywords))
                    
                    if matches > 0:
                        section_scores[node] = matches
                        
                layer_results[layer_name] = section_scores
        
        # Combine layer results with weights
        for layer_name, scores in layer_results.items():
            weight = layer_weights.get(layer_name, 1.0)
            
            for node, score in scores.items():
                if node not in fusion_scores:
                    fusion_scores[node] = 0
                
                fusion_scores[node] += score * weight
        
        # Normalize and format results
        max_score = max(fusion_scores.values()) if fusion_scores else 1.0
        
        # Return normalized results
        ranked_results = [
            {
                'node': node,
                'score': score / max_score,
                'layer_scores': {
                    layer: layer_results[layer].get(node, 0) * layer_weights.get(layer, 0)
                    for layer in layer_results if node in layer_results[layer]
                }
            }
            for node, score in fusion_scores.items()
        ]
        
        return sorted(ranked_results, key=lambda x: x['score'], reverse=True)

class InfoRetriever:
    """
    Responsible for retrieving relevant documents or knowledge base entries based on queries.
    Orchestrates the retrieval process using specialized processor classes.
    Supports vector-based, graph-based, symbolic, and hybrid retrieval methods.
    """
    def __init__(
        self,
        generator: Generator,
        knowledge_base_path: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        use_hybrid: bool = True,
        hyde_template: Optional[str] = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        local_models_dir: str = "models"
    ):
        """
        Initialize the Retriever.
        
        Args:
            generator: Generator instance for embeddings and completions
            knowledge_base_path: Path to knowledge base files
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score for results
            use_hybrid: Whether to use hybrid retrieval (both vector and graph)
            hyde_template: Template for HyDE hypothetical document generation
            reranker_model: HuggingFace model ID for reranking
            local_models_dir: Directory to store downloaded models
        """
        logger.debug(f"Initializing Retriever with knowledge_base_path={knowledge_base_path}, top_k={top_k}, threshold={similarity_threshold}")
        start_time = time.time()
        
        # Core configuration
        self.generator = generator
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_hybrid = use_hybrid
        self.local_models_dir = Path(local_models_dir)
        self.local_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        logger.debug("Initializing Retriever components")
        
        self.query_processor = QueryProcessor(generator)
        logger.debug("Query processor initialized")
        
        self.retriever_processor = RetrieverProcessor(
            generator=generator,
            knowledge_base_path=knowledge_base_path
        )
        logger.debug("Retriever processor initialized")
        
        # Set custom HyDE template if provided
        if hyde_template:
            logger.debug("Using custom HyDE template")
            self.retriever_processor.hyde_template = hyde_template
            
        self.rerank_processor = RerankProcessor(reranker_model=reranker_model)
        logger.debug("Rerank processor initialized")
        
        self.evaluation_processor = EvaluationProcessor(generator=generator)
        logger.debug("Evaluation processor initialized")
        
        logger.debug(f"Retriever initialization completed in {time.time() - start_time:.2f} seconds with {use_hybrid and 'hybrid' or 'vector-only'} retrieval mode")
    

    def vector_retrieval(self, query: str, corpus: List[Dict[str, Any]], top_k: int = None, top_p: float = None):
        logger.debug(f"Starting vector retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}' with corpus size: {len(corpus)}")
        start_time = time.time()
        
        # TO DO: implement vector retrieval
        
        logger.debug(f"Vector retrieval completed in {time.time() - start_time:.2f} seconds")
        return None

    def graph_retrieval(self, query: str, corpus: List[Dict[str, Any]], top_k: int = None, top_p: float = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.debug(f"Starting graph retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}' with corpus size: {len(corpus)}")
        start_time = time.time()
        
        # TO DO: implement graph retrieval
        
        logger.debug(f"Graph retrieval completed in {time.time() - start_time:.2f} seconds")
        return None

    # so far, not in the plan as no value
    def hybrid_retrieval(self, query: str, corpus: List[Dict[str, Any]], top_k: int = None, top_p: float = None):
        logger.debug(f"Starting hybrid retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}' with corpus size: {len(corpus)}")
        start_time = time.time()
        
        # TO DO: implement hybrid retrieval
        
        logger.debug(f"Hybrid retrieval completed in {time.time() - start_time:.2f} seconds")
        return None
