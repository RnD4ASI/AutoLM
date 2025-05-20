import json
import os
import re
import uuid
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd

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
    # In contextualiser.py:
    def persist_memory(self):
        """Save all memory types to persistent storage."""
        # Ensure directories exist
        os.makedirs("db/memory/task", exist_ok=True)
        os.makedirs("db/memory/goal", exist_ok=True)
        os.makedirs("db/memory/procedural", exist_ok=True)
        os.makedirs("db/memory/meta", exist_ok=True)
        
        # Convert and save each memory type
        pd.DataFrame(list(self.short_term_task_memory.values())).to_parquet("db/memory/task/task_memory.parquet")
        pd.DataFrame(list(self.short_term_goal_memory.values())).to_parquet("db/memory/goal/goal_memory.parquet")
        pd.DataFrame(list(self.procedural_memory.values())).to_parquet("db/memory/procedural/proc_memory.parquet")
        pd.DataFrame(list(self.meta_memory.values())).to_parquet("db/memory/meta/meta_memory.parquet")
    def store_task_memory(self, 
                         task_id: str, 
                         original_query: str, 
                         retrieved_info: Dict[str, Any],
                         transformed_query: Optional[str] = None) -> str:
        """Store information in short-term task memory."""
        memory_id = str(uuid.uuid4())
        
        # Load schema definition
        schema_path = Path('db/schema/memory_db_schema.json')
        try:
            schema = self.datautility.text_operation('load', schema_path, file_type='json')
            task_memory_schema = schema.get('collections', {}).get('short_term_task_memory', {})
            
            # Create memory entry based on schema
            memory_entry = {}
            
            # Add required fields
            required_fields = task_memory_schema.get('required_fields', {})
            for field_name in required_fields:
                if field_name == 'memory_id':
                    memory_entry[field_name] = memory_id
                elif field_name == 'task_id':
                    memory_entry[field_name] = task_id
                elif field_name == 'original_query':
                    memory_entry[field_name] = original_query
                elif field_name == 'retrieved_info':
                    memory_entry[field_name] = retrieved_info
            
            # Add optional fields if provided
            optional_fields = task_memory_schema.get('optional_fields', {})
            for field_name in optional_fields:
                if field_name == 'transformed_query' and transformed_query is not None:
                    memory_entry[field_name] = transformed_query
            
            logger.debug(f"Created memory entry using schema: {list(memory_entry.keys())}")
            
        except Exception as e:
            # Fallback to hardcoded structure if schema cannot be loaded
            logger.warning(f"Failed to load memory schema, using hardcoded structure: {e}")
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
        
        # Load schema definition
        schema_path = Path('db/schema/memory_db_schema.json')
        try:
            schema = self.datautility.text_operation('load', schema_path, file_type='json')
            goal_memory_schema = schema.get('collections', {}).get('short_term_goal_memory', {})
            
            # Create memory entry based on schema
            memory_entry = {}
            
            # Add required fields
            required_fields = goal_memory_schema.get('required_fields', {})
            for field_name in required_fields:
                if field_name == 'entity_id':
                    memory_entry[field_name] = entity_id
                elif field_name == 'entity_name':
                    memory_entry[field_name] = entity_name
                elif field_name == 'entity_information':
                    memory_entry[field_name] = entity_information
                elif field_name == 'information_type':
                    # Validate information_type against schema enum if available
                    valid_types = required_fields.get('information_type', {}).get('enum', [])
                    if valid_types and information_type not in valid_types:
                        logger.warning(f"Information type '{information_type}' not in schema enum {valid_types}, using anyway")
                    memory_entry[field_name] = information_type
            
            # Add optional fields if provided
            optional_fields = goal_memory_schema.get('optional_fields', {})
            for field_name in optional_fields:
                if field_name == 'goal_context' and goal_context is not None:
                    memory_entry[field_name] = goal_context
            
            logger.debug(f"Created goal memory entry using schema: {list(memory_entry.keys())}")
            
        except Exception as e:
            # Fallback to hardcoded structure if schema cannot be loaded
            logger.warning(f"Failed to load memory schema, using hardcoded structure: {e}")
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
        
        # Load schema definition
        schema_path = Path('db/schema/memory_db_schema.json')
        try:
            schema = self.datautility.text_operation('load', schema_path, file_type='json')
            procedural_schema = schema.get('collections', {}).get('procedural_memory', {})
            
            # Create memory entry based on schema
            memory_entry = {}
            
            # Add required fields
            required_fields = procedural_schema.get('required_fields', {})
            for field_name in required_fields:
                if field_name == 'config_id':
                    memory_entry[field_name] = config_id
                elif field_name == 'config_name':
                    memory_entry[field_name] = config_name
                elif field_name == 'config_type':
                    # Validate config_type against schema enum if available
                    valid_types = required_fields.get('config_type', {}).get('enum', [])
                    if valid_types and config_type not in valid_types:
                        logger.warning(f"Config type '{config_type}' not in schema enum {valid_types}, using anyway")
                    memory_entry[field_name] = config_type
                elif field_name == 'config_content':
                    memory_entry[field_name] = config_content
            
            # Add optional fields if provided
            optional_fields = procedural_schema.get('optional_fields', {})
            for field_name in optional_fields:
                # Handle any optional fields here if needed
                pass
            
            logger.debug(f"Created procedural memory entry using schema: {list(memory_entry.keys())}")
            
        except Exception as e:
            # Fallback to hardcoded structure if schema cannot be loaded
            logger.warning(f"Failed to load memory schema, using hardcoded structure: {e}")
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
