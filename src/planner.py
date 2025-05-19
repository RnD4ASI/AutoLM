import json
import time
import traceback
from src.logging import get_logger
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

from src.generator import Generator, MetaGenerator
from src.utility import AIUtility, DataUtility
from src.evaluator import Evaluator

logger = get_logger(__name__)

class TaskPlanner:
    """
    Responsible for planning task execution by selecting and sequencing relevant task prompts
    based on a user-provided goal and the task prompt library.
    """
    
    def __init__(self, 
                 config_file_path: Optional[Union[str, Path]] = None,
                 generator: Optional[Generator] = None):
        """
        Initialize the TaskPlanner.
        
        Args:
            config_file_path: Path to the configuration file. If None, uses default path.
            generator: Optional Generator instance. If None, creates a new one.
        """
        logger.debug("TaskPlanner initialization started")
        try:
            start_time = time.time()
            
            # Initialize utilities and components
            self.generator = generator if generator else Generator()
            self.metagenerator = MetaGenerator(generator=self.generator)
            self.aiutility = AIUtility()
            self.datautility = DataUtility()
            self.evaluator = Evaluator(generator=self.generator)
            
            # Load configuration
            config_path = Path(config_file_path) if config_file_path else Path.cwd() / "config" / "main_config.json"
            try:
                self.config = self.datautility.text_operation('load', config_path, file_type='json')
                logger.debug(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
                self.config = {}
            
            # Planning configuration
            self.default_max_prompts = 10
            self.default_similarity_threshold = 0.3
            self.default_temperature = 0.7
            self.default_model = "Qwen2.5-1.5B"
            self.default_embedding_model = "Jina-embeddings-v3"
            
            logger.debug(f"TaskPlanner initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"TaskPlanner initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def plan_task_sequence(self,
                          goal: str,
                          task_prompt_library_path: Optional[Union[str, Path]] = None,
                          max_prompts: Optional[int] = None,
                          similarity_threshold: Optional[float] = None,
                          model: Optional[str] = None,
                          embedding_model: Optional[str] = None,
                          temperature: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Plan task sequence by selecting and ordering relevant prompts based on the goal.
        
        Args:
            goal: User-provided goal description
            task_prompt_library_path: Path to task prompt library JSON file
            max_prompts: Maximum number of prompts to select
            similarity_threshold: Minimum similarity threshold for prompt selection
            model: Model to use for meta-generation calls
            embedding_model: Model to use for embeddings
            temperature: Temperature for meta-generation calls
            
        Returns:
            Tuple containing:
                - prompt_flow_config: Dictionary in prompt_flow_config format
                - shortlisted_prompts: Dictionary of selected prompts in task_prompt_library format
        """
        logger.info(f"Starting task sequence planning for goal: '{goal[:100]}{'...' if len(goal) > 100 else ''}'")
        start_time = time.time()
        
        # Set defaults
        max_prompts = max_prompts or self.default_max_prompts
        similarity_threshold = similarity_threshold or self.default_similarity_threshold
        model = model or self.default_model
        embedding_model = embedding_model or self.default_embedding_model
        temperature = temperature or self.default_temperature
        
        try:
            # Step 1: Load task prompt library
            task_library = self._load_task_prompt_library(task_prompt_library_path)
            logger.info(f"Loaded task prompt library with {len(task_library)} prompts")
            
            # Step 2: Select relevant prompts based on goal
            selected_prompts = self._select_relevant_prompts(
                goal=goal,
                task_library=task_library,
                max_prompts=max_prompts,
                similarity_threshold=similarity_threshold,
                embedding_model=embedding_model
            )
            logger.info(f"Selected {len(selected_prompts)} relevant prompts")
            
            # Step 3: Sequence the selected prompts
            sequenced_prompts = self._sequence_prompts(
                goal=goal,
                selected_prompts=selected_prompts,
                model=model,
                temperature=temperature
            )
            logger.info(f"Sequenced {len(sequenced_prompts)} prompts")
            
            # Step 4: Generate prompt flow configuration
            prompt_flow_config = self._generate_prompt_flow_config(sequenced_prompts)
            
            # Step 5: Create shortlisted prompts dictionary
            shortlisted_prompts = self._create_shortlisted_prompts(sequenced_prompts, task_library)
            
            planning_time = time.time() - start_time
            logger.info(f"Task sequence planning completed in {planning_time:.2f} seconds")
            
            return prompt_flow_config, shortlisted_prompts
            
        except Exception as e:
            logger.error(f"Task sequence planning failed: {str(e)}")
            logger.debug(f"Planning error details: {traceback.format_exc()}")
            raise
    
    def _load_task_prompt_library(self, library_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load task prompt library from JSON file."""
        if library_path is None:
            library_path = Path.cwd() / "config" / "task_prompt_library.json"
        else:
            library_path = Path(library_path)
        
        try:
            task_library = self.datautility.text_operation('load', library_path, file_type='json')
            logger.debug(f"Successfully loaded task library from {library_path}")
            return task_library
        except Exception as e:
            logger.error(f"Failed to load task prompt library from {library_path}: {e}")
            raise
    
    def _select_relevant_prompts(self,
                                goal: str,
                                task_library: Dict[str, Any],
                                max_prompts: int,
                                similarity_threshold: float,
                                embedding_model: str) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Select relevant prompts based on semantic similarity to the goal.
        Uses evaluator's BERTScore functionality for consistent similarity calculation.
        
        Returns:
            List of tuples (prompt_id, prompt_data, similarity_score)
        """
        logger.debug(f"Selecting relevant prompts with similarity threshold {similarity_threshold}")
        
        try:
            # Calculate similarities for each prompt using evaluator's BERTScore
            prompt_similarities = []
            
            for prompt_id, prompt_data in task_library.items():
                # Create searchable text from prompt
                searchable_text = self._create_searchable_text(prompt_data)
                
                # Use evaluator's BERTScore for consistent similarity calculation
                similarity = self.evaluator.get_bertscore(
                    reference_text=goal,
                    generated_text=searchable_text,
                    granularity='document',
                    model=embedding_model,
                    mode='calculation'
                )
                
                prompt_similarities.append((prompt_id, prompt_data, float(similarity)))
                logger.debug(f"Prompt {prompt_id}: similarity = {similarity:.4f}")
            
            # Filter by threshold and sort by similarity
            relevant_prompts = [
                (pid, pdata, sim) for pid, pdata, sim in prompt_similarities
                if sim >= similarity_threshold
            ]
            relevant_prompts.sort(key=lambda x: x[2], reverse=True)
            
            # Limit to max_prompts
            selected_prompts = relevant_prompts[:max_prompts]
            
            logger.info(f"Selected {len(selected_prompts)} prompts above similarity threshold {similarity_threshold}")
            
            return selected_prompts
            
        except Exception as e:
            logger.error(f"Failed to select relevant prompts: {e}")
            raise
    
    def _create_searchable_text(self, prompt_data: Dict[str, Any]) -> str:
        """Create searchable text from prompt data for similarity calculation."""
        searchable_parts = []
        
        # Add description
        if 'description' in prompt_data:
            searchable_parts.append(prompt_data['description'])
        
        # Add index keywords
        if 'index' in prompt_data:
            if isinstance(prompt_data['index'], list):
                searchable_parts.extend(prompt_data['index'])
            else:
                searchable_parts.append(str(prompt_data['index']))
        
        # Add components
        if 'components' in prompt_data:
            components = prompt_data['components']
            for key, value in components.items():
                if key in ['task', 'purpose', 'role', 'context']:
                    searchable_parts.append(str(value))
        
        return ' '.join(searchable_parts)
    
    def _sequence_prompts(self,
                         goal: str,
                         selected_prompts: List[Tuple[str, Dict[str, Any], float]],
                         model: str,
                         temperature: float) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Sequence the selected prompts based on logical dependencies and goal requirements.
        Uses meta-generation for intelligent sequencing.
        """
        logger.debug(f"Sequencing {len(selected_prompts)} selected prompts")
        
        try:
            # Prepare prompt information for meta-generation
            prompts_info = []
            for prompt_id, prompt_data, similarity in selected_prompts:
                prompt_info = {
                    'prompt_id': prompt_id,
                    'description': prompt_data.get('description', ''),
                    'task_type': self._infer_task_type(prompt_data),
                    'similarity': round(similarity, 3)
                }
                
                # Add task details if available
                if 'components' in prompt_data:
                    components = prompt_data['components']
                    prompt_info['task'] = components.get('task', '')
                    prompt_info['purpose'] = components.get('purpose', '')
                
                prompts_info.append(prompt_info)
            
            # Use meta-generation to determine optimal sequence
            try:
                sequence_response = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="decompose",
                    prompt_id=999,  # Using a default ID for planning
                    task_prompt=f"""Goal: {goal}

Available task prompts: {json.dumps(prompts_info, indent=2)}

Determine the optimal sequence for executing these task prompts to achieve the goal.
Consider:
1. Logical dependencies (which tasks need outputs from others)
2. Information flow requirements
3. Complexity progression (simple to complex)
4. Goal achievement strategy

Return your response in this JSON format:
{{
    "reasoning": "explanation of the sequencing logic",
    "sequence": ["prompt_id_1", "prompt_id_2", "prompt_id_3", ...]
}}""",
                    model=model,
                    temperature=temperature,
                    json_schema={
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "sequence": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["reasoning", "sequence"]
                    },
                    return_full_response=False
                )
                
                # Parse response
                if isinstance(sequence_response, str):
                    sequence_data = self.aiutility.format_json_response(sequence_response)
                else:
                    sequence_data = sequence_response
                
                sequence_order = sequence_data.get('sequence', [])
                reasoning = sequence_data.get('reasoning', 'Not provided')
                
                logger.debug(f"Generated sequence: {sequence_order}")
                logger.debug(f"Reasoning: {reasoning}")
                
            except Exception as e:
                logger.warning(f"Meta-generation sequencing failed: {e}, using heuristic-based ordering")
                # Fallback: use task type and similarity based ordering
                sequence_order = self._heuristic_sequencing(selected_prompts)
            
            # Create sequenced list maintaining prompt data
            prompt_dict = {pid: (pid, pdata) for pid, pdata, _ in selected_prompts}
            sequenced_prompts = []
            
            # Add prompts in the determined sequence
            for prompt_id in sequence_order:
                if prompt_id in prompt_dict:
                    sequenced_prompts.append(prompt_dict[prompt_id])
            
            # Add any remaining prompts not in the sequence
            used_ids = set(sequence_order)
            for prompt_id, prompt_data, _ in selected_prompts:
                if prompt_id not in used_ids:
                    sequenced_prompts.append((prompt_id, prompt_data))
            
            logger.info(f"Successfully sequenced {len(sequenced_prompts)} prompts")
            return sequenced_prompts
            
        except Exception as e:
            logger.error(f"Failed to sequence prompts: {e}")
            # Fallback: return in similarity order
            return [(pid, pdata) for pid, pdata, _ in selected_prompts]
    
    def _heuristic_sequencing(self, selected_prompts: List[Tuple[str, Dict[str, Any], float]]) -> List[str]:
        """
        Fallback heuristic sequencing based on task types and similarity.
        Follows typical workflow patterns: data gathering -> analysis -> synthesis -> documentation.
        """
        logger.debug("Using heuristic sequencing as fallback")
        
        # Define task type priorities (lower number = earlier in sequence)
        task_type_priorities = {
            'entity_detection': 1,      # Extract entities first
            'classification': 2,        # Classify/categorize 
            'clustering': 3,           # Group related items
            'deduction': 4,            # Analyze and reason
            'induction': 5,            # Synthesize patterns
            'documentation': 6          # Document results
        }
        
        # Create tuples of (priority, similarity, prompt_id) for sorting
        sortable_prompts = []
        for prompt_id, prompt_data, similarity in selected_prompts:
            task_type = self._infer_task_type(prompt_data)
            priority = task_type_priorities.get(task_type, 5)  # Default to middle priority
            sortable_prompts.append((priority, -similarity, prompt_id))  # Negative similarity for desc order
        
        # Sort by priority first, then by similarity (descending)
        sortable_prompts.sort()
        
        # Return ordered prompt IDs
        return [prompt_id for _, _, prompt_id in sortable_prompts]
    
    def _generate_prompt_flow_config(self, sequenced_prompts: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate prompt flow configuration in the required format."""
        logger.debug("Generating prompt flow configuration")
        
        try:
            # Create nodes
            nodes = {}
            edges = []
            
            for i, (prompt_id, prompt_data) in enumerate(sequenced_prompts):
                node_id = f"node_{i + 1}"
                
                # Extract task type from prompt data or infer from description
                task_type = self._infer_task_type(prompt_data)
                
                # Create task summary from description or task component
                task_summary = prompt_data.get('description', '')
                if not task_summary and 'components' in prompt_data:
                    task_summary = prompt_data['components'].get('task', f"Execute task from {prompt_id}")
                
                # Extract promptid or generate one
                promptid = prompt_data.get('promptid')
                if promptid is None:
                    # Try to extract from prompt_id (e.g., "prompt_001" -> 1)
                    try:
                        promptid = int(prompt_id.split('_')[-1])
                    except (ValueError, IndexError):
                        promptid = i + 1
                
                nodes[node_id] = {
                    "task_prompt_id": promptid,
                    "task_summary": task_summary,
                    "task_type": task_type
                }
                
                # Create edges (sequential flow with some intelligent branching)
                if i > 0:
                    edges.append({
                        "source": f"node_{i}",
                        "target": node_id
                    })
            
            # Optimize edges for better workflow (add parallel execution where possible)
            edges = self._optimize_edges(nodes, edges)
            
            prompt_flow_config = {
                "nodes": nodes,
                "edges": edges
            }
            
            logger.info(f"Generated prompt flow config with {len(nodes)} nodes and {len(edges)} edges")
            return prompt_flow_config
            
        except Exception as e:
            logger.error(f"Failed to generate prompt flow config: {e}")
            raise
    
    def _infer_task_type(self, prompt_data: Dict[str, Any]) -> str:
        """Infer task type from prompt data using keyword matching."""
        # Combine all searchable text
        searchable_text = self._create_searchable_text(prompt_data).lower()
        
        # Define task type keywords with priorities
        task_type_patterns = [
            ('entity_detection', ['entity', 'extraction', 'ner', 'named entity', 'entity detection', 'extract']),
            ('classification', ['classification', 'classify', 'categorize', 'category', 'class']),
            ('clustering', ['clustering', 'cluster', 'group', 'grouping', 'segment']),
            ('documentation', ['documentation', 'document', 'api', 'technical writing', 'write', 'create']),
            ('deduction', ['review', 'analysis', 'analyze', 'evaluate', 'assess', 'audit', 'examine', 'deduction', 'logic', 'reasoning']),
            ('induction', ['synthesis', 'synthesize', 'pattern', 'generalization', 'induction', 'insights', 'trends'])
        ]
        
        # Find the best match
        for task_type, keywords in task_type_patterns:
            if any(keyword in searchable_text for keyword in keywords):
                return task_type
        
        # Default to deduction if no clear match
        return 'deduction'
    
    def _optimize_edges(self, nodes: Dict[str, Any], edges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Optimize edge connections for better parallelization where possible."""
        # Find nodes that can potentially run in parallel
        parallel_candidates = []
        sequential_nodes = []
        
        for node_id, node_data in nodes.items():
            task_type = node_data.get('task_type')
            
            # Documentation and classification tasks often can run in parallel
            if task_type in ['documentation', 'classification']:
                parallel_candidates.append(node_id)
            else:
                sequential_nodes.append(node_id)
        
        # If we have multiple parallel candidates and they're not the final nodes,
        # create parallel branches that converge later
        if len(parallel_candidates) > 1 and len(sequential_nodes) > 0:
            # Find where parallel branches should converge (next sequential node)
            convergence_nodes = [node for node in sequential_nodes 
                               if any(int(node.split('_')[1]) > int(pc.split('_')[1]) 
                                     for pc in parallel_candidates)]
            
            if convergence_nodes:
                convergence_node = min(convergence_nodes, key=lambda x: int(x.split('_')[1]))
                
                # Remove direct sequential edges between parallel candidates
                edges_to_remove = []
                for edge in edges:
                    if (edge['source'] in parallel_candidates and 
                        edge['target'] in parallel_candidates):
                        edges_to_remove.append(edge)
                
                for edge in edges_to_remove:
                    edges.remove(edge)
                
                # Add convergence edges from parallel candidates to convergence node
                for pc in parallel_candidates:
                    if not any(edge['source'] == pc and edge['target'] == convergence_node 
                             for edge in edges):
                        edges.append({
                            "source": pc,
                            "target": convergence_node
                        })
        
        return edges
    
    def _create_shortlisted_prompts(self,
                                   sequenced_prompts: List[Tuple[str, Dict[str, Any]]],
                                   task_library: Dict[str, Any]) -> Dict[str, Any]:
        """Create shortlisted prompts dictionary in task_prompt_library format."""
        logger.debug("Creating shortlisted prompts dictionary")
        
        try:
            shortlisted_prompts = {}
            
            for prompt_id, prompt_data in sequenced_prompts:
                # Ensure the prompt data maintains the original format from task library
                shortlisted_prompts[prompt_id] = prompt_data.copy()
            
            logger.info(f"Created shortlisted prompts with {len(shortlisted_prompts)} entries")
            return shortlisted_prompts
            
        except Exception as e:
            logger.error(f"Failed to create shortlisted prompts: {e}")
            raise
    
    def save_planning_results(self,
                             prompt_flow_config: Dict[str, Any],
                             shortlisted_prompts: Dict[str, Any],
                             output_dir: Optional[Union[str, Path]] = None,
                             flow_filename: str = "prompt_flow_config.json",
                             prompts_filename: str = "shortlisted_prompts.json") -> Tuple[Path, Path]:
        """Save planning results to JSON files."""
        logger.info("Saving planning results to files")
        
        try:
            # Set output directory
            if output_dir is None:
                output_dir = Path.cwd() / "output"
            else:
                output_dir = Path(output_dir)
            
            # Create output directory if it doesn't exist
            self.datautility.ensure_directory(output_dir)
            
            # Save prompt flow config
            flow_path = output_dir / flow_filename
            self.datautility.text_operation('save', flow_path, prompt_flow_config, file_type='json', indent=2)
            
            # Save shortlisted prompts
            prompts_path = output_dir / prompts_filename
            self.datautility.text_operation('save', prompts_path, shortlisted_prompts, file_type='json', indent=2)
            
            logger.info(f"Saved planning results to {flow_path} and {prompts_path}")
            return flow_path, prompts_path
            
        except Exception as e:
            logger.error(f"Failed to save planning results: {e}")
            raise
    
    def analyze_goal_complexity(self, goal: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the complexity of the goal to help with planning decisions."""
        logger.debug(f"Analyzing goal complexity for: '{goal[:100]}{'...' if len(goal) > 100 else ''}'")
        
        try:
            model = model or self.default_model
            
            # Use meta-generation to analyze goal complexity
            try:
                complexity_analysis = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="evaluation",
                    action="disambiguate",  # Using disambiguate as a proxy for complexity analysis
                    prompt_id=998,
                    task_prompt=goal,
                    model=model,
                    temperature=0.3,
                    return_full_response=False
                )
            except Exception as e:
                logger.warning(f"Meta-generation complexity analysis failed: {e}")
                complexity_analysis = self._basic_complexity_analysis(goal)
            
            # Calculate additional metrics
            word_count = len(goal.split())
            sentence_count = len([s for s in goal.split('.') if s.strip()])
            
            return {
                'analysis': complexity_analysis,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'estimated_tasks': min(max(word_count // 20, 1), 10),  # Rough estimate
                'complexity_score': self._calculate_complexity_score(word_count, sentence_count)
            }
            
        except Exception as e:
            logger.warning(f"Goal complexity analysis failed: {e}")
            return self._basic_complexity_analysis(goal)
    
    def _basic_complexity_analysis(self, goal: str) -> Dict[str, Any]:
        """Provide basic complexity analysis as fallback."""
        word_count = len(goal.split())
        sentence_count = len([s for s in goal.split('.') if s.strip()])
        
        if word_count < 10:
            analysis = "Simple goal with clear objective"
            complexity_level = "low"
        elif word_count < 30:
            analysis = "Moderate complexity goal requiring multiple steps"
            complexity_level = "medium"
        else:
            analysis = "Complex goal requiring detailed planning and multiple interconnected tasks"
            complexity_level = "high"
        
        return {
            'analysis': analysis,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'estimated_tasks': min(max(word_count // 20, 1), 10),
            'complexity_score': self._calculate_complexity_score(word_count, sentence_count),
            'complexity_level': complexity_level
        }
    
    def _calculate_complexity_score(self, word_count: int, sentence_count: int) -> float:
        """Calculate a normalized complexity score (0-1) based on goal characteristics."""
        # Base complexity on word count and sentence structure
        word_score = min(word_count / 100, 1.0)  # Normalize to 100 words max
        structure_score = min(sentence_count / 10, 1.0)  # Normalize to 10 sentences max
        
        # Weighted combination
        complexity_score = (0.7 * word_score) + (0.3 * structure_score)
        return round(complexity_score, 3)
    
    def evaluate_prompt_relevance(self, 
                                  goal: str, 
                                  prompt_data: Dict[str, Any],
                                  embedding_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate how relevant a specific prompt is to the goal.
        Uses evaluator for consistent similarity measurement.
        """
        try:
            embedding_model = embedding_model or self.default_embedding_model
            searchable_text = self._create_searchable_text(prompt_data)
            
            # Use evaluator's BERTScore for semantic similarity
            semantic_score = self.evaluator.get_bertscore(
                reference_text=goal,
                generated_text=searchable_text,
                granularity='document',
                model=embedding_model,
                mode='calculation'
            )
            
            # Use evaluator's ROUGE for lexical overlap
            rouge_scores = self.evaluator.get_rouge(
                reference_text=goal,
                generated_text=searchable_text,
                mode='calculation'
            )
            
            return {
                'semantic_similarity': float(semantic_score),
                'lexical_overlap': rouge_scores,
                'task_type': self._infer_task_type(prompt_data),
                'relevance_score': float(semantic_score)  # Primary relevance metric
            }
            
        except Exception as e:
            logger.warning(f"Prompt relevance evaluation failed: {e}")
            return {
                'semantic_similarity': 0.0,
                'lexical_overlap': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0},
                'task_type': self._infer_task_type(prompt_data),
                'relevance_score': 0.0
            }