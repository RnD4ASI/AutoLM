import json
import uuid
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

from src.logging import get_logger
from src.utility import AIUtility, DataUtility, StatisticsUtility
from src.generator import Generator, MetaGenerator
from src.evaluator import Evaluator
from src.dbbuilder import TextParser, TextChunker, VectorBuilder, GraphBuilder
from src.retriever import VectorDBRetrievalProcessor, GraphDBRetrievalProcessor, QueryProcessor, InfoRetriever
from src.topologist import PromptTopology, ScalingTopology
from src.editor import TemplateAdopter

logger = get_logger(__name__)


class ContextManager:
    """
    Manages context identification, retrieval, storage, and knowledge base operations.
    Combines memory management and knowledge base functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ContextManager with configuration."""
        self.config = config
        self.datautility = DataUtility()
        
        # Memory storage (in-memory for now, can be extended to actual database)
        self.short_term_task_memory = {}
        self.short_term_goal_memory = {}
        self.procedural_memory = {}
        self.meta_memory = {}
        
        # Initialize components for knowledge base
        self.generator = Generator()
        self.metagenerator = MetaGenerator(generator=self.generator)
        self.evaluator = Evaluator(generator=self.generator)
        self.aiutility = AIUtility()
        
        # Initialize database builders
        self.text_parser = TextParser()
        self.text_chunker = TextChunker()
        self.vector_builder = VectorBuilder(
            parser=self.text_parser, 
            chunker=self.text_chunker,
            generator=self.generator
        )
        self.graph_builder = None  # Will be initialized after vector DB creation
        
        # Database paths
        self.db_dir = Path.cwd() / "db"
        self.db_dir.mkdir(exist_ok=True)
        
        # Knowledge base configuration
        kb_config = self.config.get('knowledge_base', {})
        self.chunking_method = kb_config.get('chunking', {}).get('method', 'hierarchy')
        self.source_dir = Path.cwd() / kb_config.get('paths', {}).get('source', 'data')
        
        # Vector and graph DB paths (will be set after establishment)
        self.vector_db_path = None
        self.graph_db_path = None
        
        # Retrieval processors (will be initialized after DB establishment)
        self.vector_processor = None
        self.graph_processor = None
        self.query_processor = None
        
        logger.debug("ContextManager initialized")
    
    #--- Memory Operations ---#
    
    def store_task_memory(self, 
                         task_id: str, 
                         original_query: str, 
                         retrieved_info: Dict[str, Any],
                         transformed_query: Optional[str] = None) -> str:
        """Store information in short-term task memory."""
        memory_id = str(uuid.uuid4())
        
        memory_entry = {
            "memory_id": memory_id,
            "task_id": task_id,
            "original_query": original_query,
            "retrieved_info": retrieved_info,
            "transformed_query": transformed_query
        }
        
        self.short_term_task_memory[memory_id] = memory_entry
        logger.debug(f"Stored task memory entry {memory_id} for task {task_id}")
        return memory_id
    
    def store_goal_memory(self,
                         entity_name: str,
                         entity_information: Dict[str, Any],
                         information_type: str,
                         goal_context: Optional[str] = None) -> str:
        """Store information in short-term goal memory."""
        entity_id = str(uuid.uuid4())
        
        memory_entry = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "entity_information": entity_information,
            "information_type": information_type,
            "goal_context": goal_context
        }
        
        self.short_term_goal_memory[entity_id] = memory_entry
        logger.debug(f"Stored goal memory entry {entity_id} for entity {entity_name}")
        return entity_id
    
    def get_task_memory(self, task_id: str) -> List[Dict[str, Any]]:
        """Retrieve all memory entries for a specific task."""
        return [entry for entry in self.short_term_task_memory.values() 
                if entry["task_id"] == task_id]
    
    def get_goal_memory(self, entity_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve goal memory entries, optionally filtered by entity name."""
        if entity_name:
            return [entry for entry in self.short_term_goal_memory.values() 
                    if entry["entity_name"] == entity_name]
        return list(self.short_term_goal_memory.values())
    
    def store_procedural_memory(self, config_name: str, config_type: str, config_content: Dict[str, Any]) -> str:
        """Store configuration in procedural memory."""
        config_id = str(uuid.uuid4())
        
        memory_entry = {
            "config_id": config_id,
            "config_name": config_name,
            "config_type": config_type,
            "config_content": config_content
        }
        
        self.procedural_memory[config_id] = memory_entry
        logger.debug(f"Stored procedural memory entry {config_id} for {config_name}")
        return config_id
    
    def get_procedural_memory(self, config_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve procedural memory entries, optionally filtered by type."""
        if config_type:
            return [entry for entry in self.procedural_memory.values() 
                    if entry["config_type"] == config_type]
        return list(self.procedural_memory.values())
    
    #--- Knowledge Base Operations ---#
    
    def establish_knowledge_base(self, 
                                source_files: Optional[List[Union[str, Path]]] = None,
                                df_headings: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
        """
        Establish knowledge base from source files.
        
        Args:
            source_files: List of source files to process. If None, processes all files in source directory.
            df_headings: DataFrame with heading metadata for hierarchy-based chunking.
            
        Returns:
            Tuple of (vector_db_path, graph_db_path)
        """
        logger.info("Starting knowledge base establishment")
        start_time = time.time()
        
        try:
            # Determine source files to process
            if source_files is None:
                source_files = self._discover_source_files()
            else:
                source_files = [Path(f) for f in source_files]
            
            logger.info(f"Processing {len(source_files)} source files")
            
            vector_db_paths = []
            markdown_files = []
            
            # Process each source file
            for source_file in source_files:
                logger.info(f"Processing source file: {source_file}")
                
                # Convert to markdown if necessary
                if source_file.suffix.lower() == '.pdf':
                    markdown_file = self._convert_pdf_to_markdown(source_file)
                else:
                    markdown_file = source_file
                
                markdown_files.append(markdown_file)
                
                # Create vector database for this file
                vector_db_path = self._create_vector_database(markdown_file, df_headings)
                vector_db_paths.append(vector_db_path)
            
            # Merge vector databases if multiple files
            if len(vector_db_paths) > 1:
                merged_vector_db = self.vector_builder.merge_vectordbs(vector_db_paths)
                final_vector_db_path = str(self.db_dir / "merged_vector_db.parquet")
                merged_vector_db.to_parquet(final_vector_db_path)
            else:
                final_vector_db_path = vector_db_paths[0]
            
            # Create graph database
            graph_db_path = self._create_graph_database(final_vector_db_path)
            
            # Store paths and initialize retrieval processors
            self.vector_db_path = final_vector_db_path
            self.graph_db_path = graph_db_path
            
            # Initialize retrieval processors
            self._initialize_retrieval_processors()
            
            establishment_time = time.time() - start_time
            logger.info(f"Knowledge base establishment completed in {establishment_time:.2f} seconds")
            logger.info(f"Vector DB: {final_vector_db_path}")
            logger.info(f"Graph DB: {graph_db_path}")
            
            return final_vector_db_path, graph_db_path
            
        except Exception as e:
            logger.error(f"Knowledge base establishment failed: {str(e)}")
            logger.debug(f"Establishment error details: {traceback.format_exc()}")
            raise
    
    def _initialize_retrieval_processors(self):
        """Initialize retrieval processors with the established databases."""
        if self.vector_db_path and self.graph_db_path:
            self.vector_processor = VectorDBRetrievalProcessor(
                vector_file=self.vector_db_path,
                generator=self.generator
            )
            self.graph_processor = GraphDBRetrievalProcessor(
                graph_file=self.graph_db_path
            )
            self.query_processor = QueryProcessor(generator=self.generator)
            logger.debug("Retrieval processors initialized")
        else:
            logger.warning("Cannot initialize retrieval processors: database paths not set")
    
    def _discover_source_files(self) -> List[Path]:
        """Discover source files in the configured source directory."""
        if not self.source_dir.exists():
            logger.warning(f"Source directory {self.source_dir} does not exist")
            return []
        
        # Get supported file types from configuration
        supported_types = self.config.get('system', {}).get('supported_file_types', ['pdf', 'md', 'txt'])
        
        source_files = []
        for file_type in supported_types:
            if file_type in ['pdf', 'md', 'txt']:
                pattern = f"*.{file_type}"
                source_files.extend(self.source_dir.glob(pattern))
        
        logger.debug(f"Discovered {len(source_files)} source files")
        return source_files
    
    def _convert_pdf_to_markdown(self, pdf_path: Path) -> Path:
        """Convert PDF to markdown using the text parser."""
        logger.debug(f"Converting PDF to markdown: {pdf_path}")
        
        try:
            # Use markitdown as default conversion method
            self.text_parser.pdf2md_markitdown(str(pdf_path))
            markdown_path = pdf_path.with_suffix('.md')
            
            if markdown_path.exists():
                logger.debug(f"Successfully converted {pdf_path} to {markdown_path}")
                return markdown_path
            else:
                raise FileNotFoundError(f"Markdown file not created: {markdown_path}")
                
        except Exception as e:
            logger.error(f"PDF conversion failed for {pdf_path}: {e}")
            raise
    
    def _create_vector_database(self, markdown_file: Path, df_headings: Optional[pd.DataFrame] = None) -> str:
        """Create vector database from markdown file."""
        logger.debug(f"Creating vector database for: {markdown_file}")
        
        try:
            if self.chunking_method == 'hierarchy' and df_headings is not None:
                vector_db_path = self.vector_builder.create_vectordb(
                    markdown_file=str(markdown_file),
                    df_headings=df_headings,
                    chunking_method='hierarchy'
                )
            else:
                # Use length-based chunking as fallback
                kb_config = self.config.get('knowledge_base', {})
                chunk_config = kb_config.get('chunking', {}).get('length_config', {})
                
                vector_db_path = self.vector_builder.create_vectordb(
                    markdown_file=str(markdown_file),
                    df_headings=pd.DataFrame(),  # Empty DataFrame for length-based
                    chunking_method='length',
                    chunk_size=chunk_config.get('chunk_size', 1000),
                    chunk_overlap=chunk_config.get('chunk_overlap', 100)
                )
            
            logger.debug(f"Created vector database: {vector_db_path}")
            return vector_db_path
            
        except Exception as e:
            logger.error(f"Vector database creation failed for {markdown_file}: {e}")
            raise
    
    def _create_graph_database(self, vector_db_path: str) -> str:
        """Create graph database from vector database."""
        logger.debug(f"Creating graph database from: {vector_db_path}")
        
        try:
            # Load vector database
            vector_df = pd.read_parquet(vector_db_path)
            
            # Initialize graph builder with vector database
            self.graph_builder = GraphBuilder(vectordb_file=vector_db_path)
            
            # Build standard graph
            graph = self.graph_builder.build_graph(vector_df)
            
            # Save graph database
            graph_db_path = self.graph_builder.save_graphdb('standard')
            
            logger.debug(f"Created graph database: {graph_db_path}")
            return graph_db_path
            
        except Exception as e:
            logger.error(f"Graph database creation failed: {e}")
            raise
    
    #--- Context Identification and Retrieval ---#
    
    def identify_common_contexts(self, shortlisted_prompts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify common contexts required by multiple task prompts.
        
        Args:
            shortlisted_prompts: Dictionary of selected prompts
            
        Returns:
            List of common context requirements
        """
        logger.info(f"Identifying common contexts from {len(shortlisted_prompts)} prompts")
        start_time = time.time()
        
        try:
            # Extract context requirements from each prompt
            context_requirements = []
            for prompt_id, prompt_data in shortlisted_prompts.items():
                requirements = self._extract_context_requirements(prompt_id, prompt_data)
                context_requirements.extend(requirements)
            
            logger.debug(f"Extracted {len(context_requirements)} context requirements")
            
            # Find overlapping requirements (required by at least 2 prompts)
            common_contexts = self._find_common_contexts(context_requirements)
            
            identification_time = time.time() - start_time
            logger.info(f"Identified {len(common_contexts)} common contexts in {identification_time:.2f} seconds")
            
            return common_contexts
            
        except Exception as e:
            logger.error(f"Context identification failed: {str(e)}")
            logger.debug(f"Identification error details: {traceback.format_exc()}")
            raise
    
    def _extract_context_requirements(self, prompt_id: str, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract context requirements from a single prompt."""
        requirements = []
        
        try:
            # Create searchable text from prompt components
            searchable_parts = []
            if 'components' in prompt_data:
                components = prompt_data['components']
                for key in ['task', 'purpose', 'context', 'principles']:
                    if key in components:
                        searchable_parts.append(components[key])
            
            searchable_text = ' '.join(searchable_parts)
            
            # Use meta-generation to extract context requirements
            context_analysis = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="evaluation",
                action="disambiguate",
                prompt_id=1000,
                task_prompt=f"Analyze this task prompt and identify what contextual information would be needed to execute it effectively:\n\n{searchable_text}",
                model="Qwen2.5-1.5B",
                temperature=0.5,
                return_full_response=False
            )
            
            # Parse the analysis to extract specific requirements
            # This is a simplified extraction - could be enhanced with more sophisticated NLP
            requirement = {
                'prompt_id': prompt_id,
                'context_description': searchable_text,
                'analysis': context_analysis,
                'keywords': self._extract_keywords(searchable_text)
            }
            requirements.append(requirement)
            
        except Exception as e:
            logger.warning(f"Failed to extract context requirements for {prompt_id}: {e}")
        
        return requirements
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for context matching."""
        # Simple keyword extraction - could be enhanced with more sophisticated methods
        words = text.lower().split()
        # Filter out common stop words and keep meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(keywords))  # Return unique keywords
    
    def _find_common_contexts(self, context_requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find context requirements that are common to multiple prompts."""
        common_contexts = []
        
        # Group requirements by similarity
        for i, req1 in enumerate(context_requirements):
            for j, req2 in enumerate(context_requirements[i+1:], i+1):
                # Calculate similarity between requirements
                similarity = self.evaluator.get_bertscore(
                    reference_text=req1['context_description'],
                    generated_text=req2['context_description'],
                    model="Jina-embeddings-v3",
                    mode='calculation'
                )
                
                # If similarity is high, consider it a common context
                if similarity > 0.7:
                    # Check if already added to avoid duplicates
                    is_duplicate = any(
                        cc['prompt_ids'] == {req1['prompt_id'], req2['prompt_id']}
                        for cc in common_contexts
                    )
                    
                    if not is_duplicate:
                        common_context = {
                            'prompt_ids': {req1['prompt_id'], req2['prompt_id']},
                            'description': req1['context_description'],
                            'keywords': list(set(req1['keywords'] + req2['keywords'])),
                            'similarity_score': similarity
                        }
                        common_contexts.append(common_context)
        
        return common_contexts

    def retrieve_short_term_goal_memory(self, common_contexts: List[Dict[str, Any]]) -> List[str]:
        """
        Retrieve information for common contexts and store in short-term goal memory.
        
        Args:
            common_contexts: List of common context requirements
            
        Returns:
            List of memory IDs for stored contexts
        """
        logger.info(f"Retrieving information for {len(common_contexts)} common contexts")
        start_time = time.time()
        
        memory_ids = []
        
        try:
            for i, context in enumerate(common_contexts):
                logger.debug(f"Retrieving context {i+1}/{len(common_contexts)}")
                
                # Prepare query from context description and keywords
                query = f"{context['description']} {' '.join(context['keywords'])}"
                
                # Retrieve information using both vector and graph search
                vector_results, vector_metadata = self.vector_processor.semantic_search(
                    query=query,
                    corpus=[],  # Will be loaded from database
                    top_k=5,
                    top_p=0.7
                )
                
                graph_results = self.graph_processor.semantic_cluster_search(
                    query_embedding=self.generator.get_embeddings(query, model="Jina-embeddings-v3"),
                    top_k=3
                )
                
                # Combine results
                combined_info = {
                    'vector_search_results': vector_results,
                    'vector_metadata': vector_metadata,
                    'graph_search_results': graph_results,
                    'search_query': query,
                    'related_prompts': list(context['prompt_ids'])
                }
                
                # Store in short-term goal memory
                memory_id = self.store_goal_memory(
                    entity_name=f"common_context_{i}",
                    entity_information=combined_info,
                    information_type="definitional",
                    goal_context="common_context_retrieval"
                )
                
                memory_ids.append(memory_id)
                logger.debug(f"Stored context in goal memory with ID {memory_id}")
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved and stored {len(memory_ids)} contexts in {retrieval_time:.2f} seconds")
            
            return memory_ids
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            logger.debug(f"Retrieval error details: {traceback.format_exc()}")
            raise

    def assess_context_sufficiency(self, task_prompt: Dict[str, Any], task_id: str) -> Tuple[bool, Optional[str]]:
        """
        Assess if the context available in memory is sufficient for the task.
        
        Args:
            task_prompt: The task prompt to execute
            task_id: Unique identifier for the task
            
        Returns:
            Tuple of (is_sufficient, additional_context_query)
        """
        logger.debug(f"Assessing context sufficiency for task {task_id}")
        
        try:
            # Get task-specific context requirements
            task_requirements = self._extract_context_requirements(task_id, task_prompt)[0]
            
            # Get available context from memory
            available_contexts = self.get_goal_memory()
            task_contexts = self.get_task_memory(task_id)
            
            # Create description of available context
            available_context_desc = []
            for context in available_contexts:
                available_context_desc.append(context['entity_information'].get('search_query', ''))
            for context in task_contexts:
                available_context_desc.append(context['original_query'])
            
            available_context_text = ' '.join(available_context_desc)
            
            # Use meta-generation to assess sufficiency
            sufficiency_assessment = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="evaluation",
                action="disambiguate",
                prompt_id=1001,
                task_prompt=f"""
                Task requirements: {task_requirements['context_description']}
                Available context: {available_context_text}
                Assess if the available context is sufficient to execute the task effectively. If not, specify what additional information is needed.""",
                model="Qwen2.5-1.5B",
                temperature=0.3,
                return_full_response=False
            )
            
            # Simple heuristic: if assessment mentions "insufficient" or "need", it's not sufficient
            is_sufficient = not any(keyword in sufficiency_assessment.lower() 
                                  for keyword in ['insufficient', 'not sufficient', 'need more', 'require additional'])
            
            additional_query = None if is_sufficient else task_requirements['context_description']
            
            logger.debug(f"Context sufficiency for task {task_id}: {'Sufficient' if is_sufficient else 'Insufficient'}")
            
            return is_sufficient, additional_query
            
        except Exception as e:
            logger.warning(f"Context sufficiency assessment failed for task {task_id}: {e}")
            # Default to insufficient to ensure additional context retrieval
            return False, task_requirements.get('context_description', 'Additional context needed')

    def retrieve_task_specific_context(self, task_id: str, query: str) -> str:
        """
        Retrieve task-specific context and store in short-term task memory.
        
        Args:
            task_id: Unique identifier for the task
            query: Query for context retrieval
            
        Returns:
            Memory ID for stored context
        """
        logger.debug(f"Retrieving task-specific context for task {task_id}")
        
        try:
            # Process query for better retrieval
            processed_query = self.query_processor.rephrase_query(
                query=query,
                model="Qwen2.5-1.5B",
                temperature=0.5
            )
            
            # Retrieve using vector search
            vector_results, vector_metadata = self.vector_processor.semantic_search(
                query=processed_query,
                corpus=[],  # Will be loaded from database
                top_k=3,
                top_p=0.8
            )
            
            # Retrieve using graph search
            graph_results = self.graph_processor.concept_hierarchy_search(
                query=processed_query,
                max_depth=2
            )
            
            # Combine results
            retrieved_info = {
                'vector_search_results': vector_results,
                'vector_metadata': vector_metadata,
                'graph_search_results': graph_results,
                'processed_query': processed_query
            }
            
            # Store in short-term task memory
            memory_id = self.store_task_memory(
                task_id=task_id,
                original_query=query,
                retrieved_info=retrieved_info,
                transformed_query=processed_query
            )
            
            logger.debug(f"Stored task-specific context in memory with ID {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Task-specific context retrieval failed for task {task_id}: {e}")
            raise


class ExecutionEngine:
    """
    Handles task execution with optimization based on mle_config.json.
    """
    
    def __init__(self, config: Dict[str, Any], context_manager: ContextManager):
        """Initialize ExecutionEngine with configuration and context manager."""
        self.config = config
        self.context_manager = context_manager
        
        # Initialize components
        self.generator = Generator()
        self.metagenerator = MetaGenerator(generator=self.generator)
        self.evaluator = Evaluator(generator=self.generator)
        self.aiutility = AIUtility()
        self.datautility = DataUtility()
        
        # Initialize topology and optimization components
        self.prompt_topology = PromptTopology(generator=self.generator)
        self.scaling_topology = ScalingTopology(generator=self.generator)
        self.template_adopter = TemplateAdopter()
        
        # Load MLE configuration
        self.mle_config = self._load_mle_config()
        
        logger.debug("ExecutionEngine initialized")
    
    def _load_mle_config(self) -> Dict[str, Any]:
        """Load MLE configuration from file."""
        try:
            config_path = Path.cwd() / "config" / "mle_config.json"
            mle_config = self.datautility.text_operation('load', config_path, file_type='json')
            logger.debug(f"Successfully loaded MLE configuration from {config_path}")
            return mle_config
        except Exception as e:
            logger.warning(f"Failed to load MLE configuration: {e}")
            # Return default configuration
            return {
                "defaults": {
                    "model_fallback_order": ["azure_openai", "anthropic", "huggingface"],
                    "topology_fallback_order": ["direct"],
                    "temperature_fallback_order": [0.7]
                },
                "llm_method_selection": {
                    "deduction": {
                        "medium": {
                            "method": {
                                "model_provider": "huggingface",
                                "model_name": "Qwen2.5-1.5B",
                                "tts_topology": "direct",
                                "parameters": {"temperature": 0.7}
                            }
                        }
                    }
                }
            }

    def execute_task(self,
                    task_id: str,
                    task_prompt: Dict[str, Any],
                    task_type: Optional[str] = None,
                    confidence_level: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a task with optimization based on mle_config.json.
        
        Args:
            task_id: Unique identifier for the task
            task_prompt: Task prompt data from prompt_task_chain.json format
            task_type: Optional override for task type (if not specified, will be inferred)
            confidence_level: Optional override for confidence level (if not specified, will be inferred)
            
        Returns:
            Dictionary containing:
                - task_id: Task identifier
                - response: Generated response
                - metadata: Execution metadata including model used, topology, etc.
                - performance_metrics: Timing and quality metrics
        """
        logger.info(f"Starting task execution for task {task_id}")
        start_time = time.time()
        
        try:
            # Step 1: Determine task type and confidence level
            if task_type is None or confidence_level is None:
                inferred_type, inferred_confidence = self._determine_task_type_and_confidence(task_prompt)
                task_type = task_type or inferred_type
                confidence_level = confidence_level or inferred_confidence
            
            logger.debug(f"Task {task_id}: type={task_type}, confidence={confidence_level}")
            
            # Step 2: Get execution configuration
            execution_config = self._get_execution_config(task_type, confidence_level)
            logger.debug(f"Selected execution config: {execution_config}")
            
            # Step 3: Prepare context from memory
            context_data = self._prepare_context(task_id, task_prompt)
            
            # Step 4: Apply prompt optimization if configured
            optimized_prompt = self._apply_prompt_optimization(task_prompt, execution_config)
            
            # Step 5: Execute with selected topology
            response, execution_metadata = self._execute_with_topology(
                task_id=task_id,
                task_prompt=optimized_prompt,
                execution_config=execution_config,
                context_data=context_data
            )
            
            # Step 6: Evaluate response quality
            quality_metrics = self._evaluate_response(optimized_prompt, response)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Prepare result
            result = {
                'task_id': task_id,
                'response': response,
                'metadata': {
                    'task_type': task_type,
                    'confidence_level': confidence_level,
                    'execution_config': execution_config,
                    'context_used': len(context_data) > 0,
                    **execution_metadata
                },
                'performance_metrics': {
                    'execution_time': execution_time,
                    'quality_metrics': quality_metrics
                }
            }
            
            logger.info(f"Task {task_id} completed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed for task {task_id}: {str(e)}")
            logger.debug(f"Execution error details: {traceback.format_exc()}")
            
            # Attempt fallback execution
            try:
                logger.info(f"Attempting fallback execution for task {task_id}")
                fallback_result = self._handle_execution_fallback(task_id, task_prompt)
                fallback_result['metadata']['fallback_used'] = True
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback execution also failed for task {task_id}: {fallback_error}")
                raise

    def _determine_task_type_and_confidence(self, task_prompt: Dict[str, Any]) -> Tuple[str, str]:
        """
        Determine task type and confidence level from task prompt.
        
        Args:
            task_prompt: Task prompt data
            
        Returns:
            Tuple of (task_type, confidence_level)
        """
        try:
            # Extract text for analysis
            analysis_text = ""
            if 'components' in task_prompt and isinstance(task_prompt['components'], dict):
                components = task_prompt['components']
                analysis_text = f"{components.get('task', '')} {components.get('purpose', '')} {components.get('context', '')}".strip()
            elif 'description' in task_prompt and isinstance(task_prompt['description'], str):
                analysis_text = task_prompt['description']
            elif 'name' in task_prompt and isinstance(task_prompt['name'], str):
                analysis_text = task_prompt['name'] # Fallback to name if description/components absent
            else:
                analysis_text = str(task_prompt) # Last resort: string representation
            
            # Attempt to get directly from task_prompt metadata if available
            task_type = task_prompt.get('type')
            confidence_level = task_prompt.get('confidence_level')
            
            # If not directly provided or one is missing, use meta-generation
            if not task_type or not confidence_level:
                try:
                    analysis_response = self.metagenerator.get_meta_generation(
                        application="metaprompt",
                        category="evaluation",
                        action="disambiguate",
                        prompt_id=1002,
                        task_prompt=f"""Analyze this task and determine:
                        1. Task type (one of: deduction, documentation, classification, clustering, induction, entity_detection)
                        2. Complexity/confidence level (one of: high, medium, low)

                        Task details: {analysis_text}

                        Return your response in JSON format:
                        {{"task_type": "<type>", "confidence_level": "<level>", "reasoning": "<explanation>"}}""",
                        model=self.mle_config.get("defaults", {}).get("metaprompt_model", "Qwen2.5-1.5B"),
                        temperature=self.mle_config.get("defaults", {}).get("metaprompt_temperature", 0.3),
                        json_schema={
                            "type": "object",
                            "properties": {
                                "task_type": {"type": "string"},
                                "confidence_level": {"type": "string"},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["task_type", "confidence_level"]
                        },
                        return_full_response=False
                    )
                    
                    if isinstance(analysis_response, str):
                        analysis_data = self.aiutility.format_json_response(analysis_response)
                    else:
                        analysis_data = analysis_response # Assuming it's already a dict
                    
                    # Fill with analysis result if the original was missing
                    if not task_type:
                        task_type = analysis_data.get('task_type', 'deduction')
                    if not confidence_level:
                        confidence_level = analysis_data.get('confidence_level', 'medium')
                    
                    logger.debug(f"Task analysis result via meta-generation: {analysis_data}")
                    
                except Exception as e_meta:
                    logger.warning(f"Meta-generation task analysis failed: {e_meta}, using heuristic approach")
                    # Use heuristic analysis as fallback
                    heuristic_task_type, heuristic_confidence_level = self._heuristic_task_analysis(analysis_text)
                    
                    # Fill with heuristic result if the original was missing
                    if not task_type:
                        task_type = heuristic_task_type
                    if not confidence_level:
                        confidence_level = heuristic_confidence_level
            
            # Define valid types and levels
            # Valid task types are derived from the keys in the mle_config's llm_method_selection
            valid_task_types = list(self.mle_config.get("llm_method_selection", {}).keys())
            if not valid_task_types: 
                # Fallback default list if mle_config doesn't define task types
                valid_task_types = [
                    "deduction", "summarization", "generation", "classification", 
                    "translation", "qa", "reasoning", "creative_writing", "coding", "chat"
                ]
                logger.warning("Could not derive valid_task_types from mle_config, using internal default list.")
            
            valid_confidence_levels = ["high", "medium", "low"]
            
            # Validate and default if necessary
            if task_type not in valid_task_types:
                logger.warning(f"Invalid task type '{task_type}' provided or inferred, defaulting to 'deduction'. Valid types: {valid_task_types}")
                task_type = 'deduction' # Default task type
            
            if confidence_level not in valid_confidence_levels:
                logger.warning(f"Invalid confidence level '{confidence_level}' provided or inferred, defaulting to 'medium'. Valid levels: {valid_confidence_levels}")
                confidence_level = 'medium' # Default confidence level
            
            logger.debug(f"Determined task type: {task_type}, confidence: {confidence_level}")
            return task_type, confidence_level
            
        except Exception as e:
            logger.error(f"Task type and confidence determination failed: {e}\n{traceback.format_exc()}")
            logger.warning("Defaulting to 'deduction' task type and 'medium' confidence due to error.")
            return 'deduction', 'medium'  # Safe defaults

    def _heuristic_task_analysis(self, analysis_text: str) -> Tuple[str, str]:
        """Fallback heuristic analysis for task type and confidence."""
        text_lower = analysis_text.lower()
        
        # Task type patterns
        if any(keyword in text_lower for keyword in ['entity', 'extract', 'ner', 'identify']):
            task_type = 'entity_detection'
        elif any(keyword in text_lower for keyword in ['classify', 'categorize', 'category']):
            task_type = 'classification'
        elif any(keyword in text_lower for keyword in ['cluster', 'group', 'segment']):
            task_type = 'clustering'
        elif any(keyword in text_lower for keyword in ['document', 'api', 'write', 'create']):
            task_type = 'documentation'
        elif any(keyword in text_lower for keyword in ['pattern', 'synthesize', 'insights']):
            task_type = 'induction'
        else:
            task_type = 'deduction'
        
        # Confidence level based on complexity indicators
        complexity_indicators = ['comprehensive', 'complex', 'detailed', 'thorough', 'multi', 'advanced']
        simplicity_indicators = ['simple', 'basic', 'straightforward', 'quick']
        
        if any(indicator in text_lower for indicator in complexity_indicators):
            confidence_level = 'high'
        elif any(indicator in text_lower for indicator in simplicity_indicators):
            confidence_level = 'low'
        else:
            confidence_level = 'medium'
        
        return task_type, confidence_level

    def _get_execution_config(self, task_type: str, confidence_level: str) -> Dict[str, Any]:
        """
        Get execution configuration based on task type and confidence level.
        
        Args:
            task_type: Type of task
            confidence_level: Confidence level (high/medium/low)
            
        Returns:
            Execution configuration dictionary
        """
        try:
            # Get configuration from mle_config
            llm_selection = self.mle_config.get('llm_method_selection', {})
            
            if task_type in llm_selection and confidence_level in llm_selection[task_type]:
                config = llm_selection[task_type][confidence_level]['method'].copy()
                logger.debug(f"Found specific config for {task_type}/{confidence_level}")
                return config
            
            # Try fallback with medium confidence
            if task_type in llm_selection and 'medium' in llm_selection[task_type]:
                config = llm_selection[task_type]['medium']['method'].copy()
                logger.warning(f"Using medium confidence fallback for {task_type}/{confidence_level}")
                return config
            
            # Try fallback with deduction task type
            if 'deduction' in llm_selection and confidence_level in llm_selection['deduction']:
                config = llm_selection['deduction'][confidence_level]['method'].copy()
                logger.warning(f"Using deduction fallback for {task_type}/{confidence_level}")
                return config
            
            # Ultimate fallback
            defaults = self.mle_config.get('defaults', {})
            config = {
                'model_provider': defaults.get('model_fallback_order', ['huggingface'])[0],
                'model_name': 'Qwen2.5-1.5B',
                'tts_topology': defaults.get('topology_fallback_order', ['direct'])[0],
                'parameters': {'temperature': defaults.get('temperature_fallback_order', [0.7])[0]}
            }
            logger.warning(f"Using ultimate fallback config for {task_type}/{confidence_level}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to get execution config: {e}")
            # Return minimal safe config
            return {
                'model_provider': 'huggingface',
                'model_name': 'Qwen2.5-1.5B',
                'tts_topology': 'direct',
                'parameters': {'temperature': 0.7}
            }
    
    def _prepare_context(self, task_id: str, task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context data from memory for task execution.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt data
            
        Returns:
            Context data dictionary
        """
        try:
            context_data = {}
            
            # Get task-specific memory
            task_memory = self.context_manager.get_task_memory(task_id)
            if task_memory:
                context_data['task_specific'] = task_memory
                logger.debug(f"Found {len(task_memory)} task-specific memory entries")
            
            # Get goal memory (common contexts)
            goal_memory = self.context_manager.get_goal_memory()
            if goal_memory:
                context_data['goal_context'] = goal_memory
                logger.debug(f"Found {len(goal_memory)} goal memory entries")
            
            # Get procedural memory
            procedural_memory = self.context_manager.get_procedural_memory()
            if procedural_memory:
                context_data['procedural'] = procedural_memory
                logger.debug(f"Found {len(procedural_memory)} procedural memory entries")
            
            return context_data
            
        except Exception as e:
            logger.warning(f"Failed to prepare context for task {task_id}: {e}")
            return {}

    def _apply_prompt_optimization(self, task_prompt: Dict[str, Any], execution_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply prompt optimization based on execution configuration.
        
        Args:
            task_prompt: Original task prompt
            execution_config: Execution configuration
            
        Returns:
            Optimized task prompt
        """
        try:
            optimization_method = execution_config.get('prompt_optimization', 'none')
            
            if optimization_method == 'none':
                logger.debug("No prompt optimization configured")
                return task_prompt
            
            logger.debug(f"Applying prompt optimization: {optimization_method}")
            
            if optimization_method == 'genetic_algorithm':
                # Use genetic algorithm topology for prompt optimization
                optimized_prompts, _ = self.prompt_topology.prompt_genetic_algorithm(
                    task_prompt=str(task_prompt),
                    prompt_id=1003,
                    num_variations=3,
                    num_evolution=2,
                    model="Qwen2.5-1.5B",
                    temperature=0.7,
                    return_full_response=True
                )
                # Parse the best optimized prompt back to dictionary format
                if optimized_prompts and len(optimized_prompts) > 0:
                    optimized_prompt = self.aiutility.format_json_response(optimized_prompts[0])
                    return optimized_prompt
            
            elif optimization_method == 'disambiguation':
                # Use disambiguation topology
                optimized_prompt, _ = self.prompt_topology.prompt_disambiguation(
                    task_prompt=str(task_prompt),
                    prompt_id=1004,
                    model="Qwen2.5-1.5B",
                    temperature=0.5,
                    return_full_response=False
                )
                # Parse back to dictionary format
                optimized_prompt = self.aiutility.format_json_response(optimized_prompt)
                return optimized_prompt
            
            elif optimization_method in ['chain_of_thought', 'tree_of_thought', 'program_synthesis', 'deep_thought']:
                # Use template adopter for reasoning enhancement
                optimized_prompt = self.template_adopter.get_prompt_transformation(
                    prompt_dict=task_prompt,
                    fix_template=optimization_method
                )
                return optimized_prompt
            
            else:
                logger.warning(f"Unknown optimization method: {optimization_method}")
                return task_prompt
                
        except Exception as e:
            logger.warning(f"Prompt optimization failed: {e}, using original prompt")
            return task_prompt
    
    def _execute_with_topology(self,
                              task_id: str,
                              task_prompt: Dict[str, Any],
                              execution_config: Dict[str, Any],
                              context_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Execute task using the specified topology.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt (potentially optimized)
            execution_config: Execution configuration
            context_data: Context data from memory
            
        Returns:
            Tuple of (response, execution_metadata)
        """
        start_time = time.time()
        
        # Extract configuration parameters
        model = execution_config.get('model', 'Qwen2.5-1.5B')
        temperature = execution_config.get('temperature', 0.7)
        topology = execution_config.get('topology', 'default')
        topology_params = execution_config.get('topology_params', {})
        
        logger.info(f"Executing task {task_id} with topology {topology}")
        
        # Prepare contextual prompt by combining task prompt with context
        contextual_prompt = self._prepare_contextual_prompt(task_prompt, context_data)
        
        # Execute with specified topology
        try:
            response = None
            performance_metrics = {}
            execution_details = {}
            
            # Track execution time
            topology_start_time = time.time()
            
            # Choose topology
            if topology == "default":
                # Direct execution without specific topology
                response = self.generator.get_completion(
                    prompt_id=task_id,
                    prompt=contextual_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                execution_details = {"type": "direct_execution"}
                
            elif topology == "prompt_disambiguation":
                # Topology 1: Prompt disambiguation
                result = self.prompt_topology.prompt_disambiguation(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                response = result[1]  # Optimized response
                execution_details = {
                    "type": "prompt_disambiguation",
                    "original_prompt": result[0][0],
                    "optimized_prompt": result[0][1],
                    "original_response": result[1][0],
                    "optimized_response": result[1][1]
                }
                
            elif topology == "genetic_algorithm":
                # Topology 2: Genetic algorithm
                num_variations = topology_params.get('num_variations', 5)
                num_evolution = topology_params.get('num_evolution', 3)
                result = self.prompt_topology.prompt_genetic_algorithm(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_variations=num_variations,
                    num_evolution=num_evolution,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                response = result[1]  # Best response
                execution_details = {
                    "type": "genetic_algorithm",
                    "prompt_variations": result[0],
                    "responses": result[1],
                    "num_variations": num_variations,
                    "num_evolution": num_evolution
                }
                
            elif topology == "best_of_n_synthesis":
                # Topology 3: Best of N synthesis
                num_variations = topology_params.get('num_variations', 3)
                result = self.scaling_topology.best_of_n_synthesis(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_variations=num_variations,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                if isinstance(result, dict):
                    response = result.get('response')
                    execution_details = {
                        "type": "best_of_n_synthesis",
                        "num_variations": num_variations,
                        "variations": result.get('variations')
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "best_of_n_synthesis",
                        "num_variations": num_variations
                    }
            
            elif topology == "best_of_n_selection":
                # Topology 4: Best of N selection
                num_variations = topology_params.get('num_variations', 3)
                selection_method = topology_params.get('selection_method', 'llm')
                response = self.scaling_topology.best_of_n_selection(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_variations=num_variations,
                    selection_method=selection_method,
                    model=model,
                    model_selector=model,  # Using same model for selection
                    temperature=temperature,
                    **topology_params
                )
                execution_details = {
                    "type": "best_of_n_selection",
                    "num_variations": num_variations,
                    "selection_method": selection_method
                }
                
            elif topology == "self_reflection":
                # Topology 5: Self-reflection
                num_iterations = topology_params.get('num_iterations', 1)
                result = self.scaling_topology.self_reflection(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_iterations=num_iterations,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                if isinstance(result, dict):
                    response = result.get('final_response')
                    execution_details = {
                        "type": "self_reflection",
                        "num_iterations": num_iterations,
                        "iterations": result.get('iterations')
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "self_reflection",
                        "num_iterations": num_iterations
                    }
                
            elif topology == "chain_of_thought":
                # Topology 6: Chain-of-thought reasoning
                template = topology_params.get('template', 'chain_of_thought')
                transformed_prompt = self.prompt_topology.prompt_reasoning(
                    task_prompt=contextual_prompt,
                    template=template
                )
                response = self.generator.get_completion(
                    prompt_id=task_id,
                    prompt=transformed_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                execution_details = {
                    "type": "chain_of_thought",
                    "template": template,
                    "transformed_prompt": transformed_prompt
                }
                
            elif topology == "multi_agent_debate":
                # Topology 8: Multi-agent debate
                num_iterations = topology_params.get('num_iterations', 2)
                model_strong = topology_params.get('model_strong', model)
                model_weak = topology_params.get('model_weak', model)
                selection_method = topology_params.get('selection_method', 'llm')
                result = self.scaling_topology.multi_agent_debate(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_iterations=num_iterations,
                    model_strong=model_strong,
                    model_weak=model_weak,
                    selection_method=selection_method,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                if isinstance(result, dict):
                    response = result.get('final_response')
                    execution_details = {
                        "type": "multi_agent_debate",
                        "num_iterations": num_iterations,
                        "model_strong": model_strong,
                        "model_weak": model_weak,
                        "iterations": result.get('iterations'),
                        "selection_method": selection_method
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "multi_agent_debate",
                        "num_iterations": num_iterations,
                        "model_strong": model_strong,
                        "model_weak": model_weak,
                        "selection_method": selection_method
                    }
            
            elif topology == "regenerative_majority_synthesis":
                # Topology: Regenerative Majority Synthesis
                num_initial_responses = topology_params.get('num_initial_responses', 3)
                num_regen_responses = topology_params.get('num_regen_responses', 3)
                cut_off_fraction = topology_params.get('cut_off_fraction', 0.5)
                synthesis_method = topology_params.get('synthesis_method', 'majority_vote')
                
                result = self.scaling_topology.regenerative_majority_synthesis(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_initial_responses=num_initial_responses,
                    num_regen_responses=num_regen_responses,
                    cut_off_fraction=cut_off_fraction,
                    synthesis_method=synthesis_method,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                
                if isinstance(result, dict):
                    response = result.get('final_response')
                    execution_details = {
                        "type": "regenerative_majority_synthesis",
                        "num_initial_responses": num_initial_responses,
                        "num_regen_responses": num_regen_responses,
                        "cut_off_fraction": cut_off_fraction,
                        "synthesis_method": synthesis_method,
                        "performance_metrics": result.get('performance_metrics')
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "regenerative_majority_synthesis",
                        "num_initial_responses": num_initial_responses,
                        "num_regen_responses": num_regen_responses,
                        "cut_off_fraction": cut_off_fraction,
                        "synthesis_method": synthesis_method
                    }
            
            else:
                # Default case for unknown topology
                logger.warning(f"Unknown topology '{topology}'. Falling back to default execution.")
                response = self.generator.get_completion(
                    prompt_id=task_id,
                    prompt=contextual_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                execution_details = {
                    "type": "default_fallback",
                    "reason": f"Unknown topology '{topology}'"
                }
            
            # Calculate execution time
            topology_time = time.time() - topology_start_time
            total_time = time.time() - start_time
            
            # Prepare performance metrics
            performance_metrics = {
                "topology_execution_time": topology_time,
                "total_execution_time": total_time,
                "model": model,
                "topology": topology
            }
            
            # Prepare execution metadata
            execution_metadata = {
                "task_id": task_id,
                "model": model,
                "temperature": temperature,
                "topology": topology,
                "execution_details": execution_details,
                "performance_metrics": performance_metrics
            }
            
            # Update iteration count in prompt_flow_config.json if needed
            self._update_task_iteration_count(task_id, task_prompt)
            
            logger.info(f"Task {task_id} executed successfully with topology '{topology}' in {total_time:.2f} seconds")
            return response, execution_metadata
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error executing task {task_id} with topology '{topology}': {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Create error metadata
            error_metadata = {
                "task_id": task_id,
                "model": model,
                "temperature": temperature,
                "topology": topology,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": error_time
            }
            
            # Return None with error metadata
            return None, error_metadata

    def _update_task_iteration_count(self, task_id: str, task_prompt: Dict[str, Any]) -> None:
        """
        Update iteration count for a task in prompt_flow_config.json.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt data
        """
        try:
            # Get the prompt flow config path
            config_path = Path.cwd() / "config" / "prompt_flow_config.json"
            
            # Load existing config if it exists
            if config_path.exists():
                prompt_flow_config = self.datautility.text_operation('load', config_path, file_type='json')
            else:
                prompt_flow_config = {"task_iterations": {}}
            
            # Initialize task_iterations if not present
            if "task_iterations" not in prompt_flow_config:
                prompt_flow_config["task_iterations"] = {}
            
            # Update iteration count for the task
            if task_id in prompt_flow_config["task_iterations"]:
                prompt_flow_config["task_iterations"][task_id] += 1
            else:
                prompt_flow_config["task_iterations"][task_id] = 1
            
            # Save updated config
            self.datautility.text_operation('save', config_path, prompt_flow_config, file_type='json')
            
            logger.debug(f"Updated iteration count for task {task_id} to {prompt_flow_config['task_iterations'][task_id]}")
            
        except Exception as e:
            logger.warning(f"Failed to update task iteration count for {task_id}: {e}")
    
    def _prepare_contextual_prompt(self, task_prompt: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """
        Prepare contextual prompt by combining task prompt with context data.
        
        Args:
            task_prompt: Task prompt data
            context_data: Context data from memory
            
        Returns:
            Contextualized prompt string
        """
        # Extract task prompt text
        prompt_text = task_prompt.get('prompt_text', '')
        
        # Prepare context sections
        context_sections = []
        
        # Add goal contexts
        goal_contexts = context_data.get('goal_contexts', {})
        if goal_contexts:
            context_str = "\n\n### Goal Context:\n"
            for entity, info in goal_contexts.items():
                context_str += f"\n## {entity}:\n{info}\n"
            context_sections.append(context_str)
        
        # Add task contexts
        task_contexts = context_data.get('task_contexts', {})
        if task_contexts:
            context_str = "\n\n### Task-Specific Context:\n"
            for entity, info in task_contexts.items():
                context_str += f"\n## {entity}:\n{info}\n"
            context_sections.append(context_str)
        
        # Add sequential context (previous task outputs)
        sequential_contexts = context_data.get('sequential_contexts', {})
        if sequential_contexts:
            context_str = "\n\n### Previous Task Outputs:\n"
            for task_id, output in sequential_contexts.items():
                context_str += f"\n## Task {task_id}:\n{output}\n"
            context_sections.append(context_str)
        
        # Combine context sections with prompt
        if context_sections:
            context_block = "\n".join(context_sections)
            contextual_prompt = f"{prompt_text}\n\n{context_block}"
        else:
            contextual_prompt = prompt_text
        
        logger.debug(f"Prepared contextual prompt with {len(context_sections)} context sections")
        return contextual_prompt
    
    def _evaluate_response(self, prompt: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate response quality.
        
        Args:
            prompt: Task prompt
            response: Generated response
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # Basic metrics
            metrics = {
                "response_length": len(response),
                "response_word_count": len(response.split())
            }
            
            # Add more sophisticated evaluation if needed
            # For example, using the evaluator component
            
            return metrics
        except Exception as e:
            logger.warning(f"Response evaluation failed: {e}")
            return {"evaluation_error": str(e)}
    
    def _handle_execution_fallback(self, task_id: str, task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle fallback execution when primary execution fails.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt data
            
        Returns:
            Fallback execution result
        """
        logger.info(f"Using fallback execution for task {task_id}")
        
        try:
            # Use simplest possible execution with most reliable model
            fallback_model = self.mle_config.get('defaults', {}).get('fallback_model', 'Qwen2.5-1.5B')
            fallback_temp = self.mle_config.get('defaults', {}).get('fallback_temperature', 0.3)
            
            # Extract prompt text
            prompt_text = ""
            if isinstance(task_prompt, dict):
                prompt_text = task_prompt.get('prompt_text', str(task_prompt))
            else:
                prompt_text = str(task_prompt)
            
            # Direct execution
            response = self.generator.get_completion(
                prompt_id=f"{task_id}_fallback",
                prompt=prompt_text,
                model=fallback_model,
                temperature=fallback_temp,
                return_full_response=False
            )
            
            # Create minimal result
            result = {
                'task_id': task_id,
                'response': response,
                'metadata': {
                    'task_type': 'unknown',
                    'confidence_level': 'low',
                    'execution_config': {
                        'model': fallback_model,
                        'temperature': fallback_temp,
                        'topology': 'direct'
                    },
                    'context_used': False
                },
                'performance_metrics': {
                    'execution_time': 0,
                    'quality_metrics': {
                        'response_length': len(response),
                        'response_word_count': len(response.split())
                    }
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback execution also failed: {e}")
            raise
    
    def orchestrate_execution_flow(self, plan: List[Dict[str, Any]], overall_goal_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Orchestrates the execution of a sequence of tasks based on a plan.

        Args:
            plan: A list of task definitions. Each task definition is a dictionary
                  expected to contain 'task_id', 'task_prompt', and optionally
                  'task_type', 'confidence_level'.
            overall_goal_context: Optional string describing the overarching goal for context.

        Returns:
            A list of dictionaries, where each dictionary contains the execution
            result for a corresponding task in the plan.
        """
        logger.info(f"Starting orchestration for a plan with {len(plan)} tasks.")
        all_results = []

        for i, task_definition in enumerate(plan):
            task_id = task_definition.get("task_id", f"task_{uuid.uuid4()}")
            task_prompt = task_definition.get("task_prompt")
            task_type = task_definition.get("task_type")
            confidence_level = task_definition.get("confidence_level")

            if not task_prompt:
                logger.warning(f"Skipping task {i+1} (ID: {task_id}) due to missing 'task_prompt'.")
                all_results.append({
                    "task_id": task_id,
                    "status": "skipped",
                    "reason": "Missing task_prompt"
                })
                continue

            logger.info(f"Executing task {i+1}/{len(plan)}: ID {task_id}")
            
            try:
                # Execute the task
                execution_result = self.execute_task(
                    task_id=task_id,
                    task_prompt=task_prompt,
                    task_type=task_type,
                    confidence_level=confidence_level
                )
                all_results.append(execution_result)

                # Store the result in memory
                if execution_result and execution_result.get("response") is not None:
                    self.context_manager.store_goal_memory(
                        entity_name=f"task_result_{task_id}",
                        entity_information=execution_result, # Store the whole result package
                        information_type="task_execution_result",
                        goal_context=overall_goal_context or f"Result of orchestrated task {task_id}"
                    )
                    logger.info(f"Stored result for task {task_id} in goal memory.")
                elif execution_result:
                    logger.warning(f"Task {task_id} executed but produced no response or an error. Result: {execution_result}")
                else:
                    logger.error(f"Task {task_id} execution returned None. Not storing in memory.")

            except Exception as e:
                logger.error(f"Critical error during orchestration of task {task_id}: {e}")
                logger.error(traceback.format_exc())
                all_results.append({
                    "task_id": task_id,
                    "response": None,
                    "metadata": {"error": f"Orchestration failure: {str(e)}"},
                    "performance_metrics": {}
                })
        
        # Update prompt_flow_config.json with completion of this orchestration
        try:
            config_path = Path.cwd() / "config" / "prompt_flow_config.json"
            if config_path.exists():
                prompt_flow_config = self.datautility.text_operation('load', config_path, file_type='json')
            else:
                prompt_flow_config = {}
            
            # Record orchestration completion
            if "orchestrations" not in prompt_flow_config:
                prompt_flow_config["orchestrations"] = []
            
            orchestration_record = {
                "timestamp": time.time(),
                "tasks_count": len(plan),
                "completed_tasks": len([r for r in all_results if r.get("response") is not None]),
                "goal_context": overall_goal_context,
                "task_ids": [task.get("task_id") for task in plan if "task_id" in task]
            }
            
            prompt_flow_config["orchestrations"].append(orchestration_record)
            self.datautility.text_operation('save', config_path, prompt_flow_config, file_type='json')
            
        except Exception as e:
            logger.warning(f"Failed to update prompt_flow_config.json with orchestration record: {e}")
        
        logger.info(f"Orchestration completed. Processed {len(all_results)} tasks.")
        return all_results


