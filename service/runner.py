"""
AutoLM System Runner

This is orchestrator that implements AutoLM workflow following the data flow design:
1. Knowledge Base Establishment
2. Goal-driven Task Planning
3. Context Retrieval and Memory Management
4. Task Execution with Optimization
5. Evaluation and Output Generation

Usage:
    python runner.py --goal "Your goal description" --config config/main_config.json
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import all necessary modules
from src.logging import get_logger
from src.utility import DataUtility, AIUtility, StatisticsUtility
from src.generator import Generator, MetaGenerator
from src.evaluator import Evaluator
from src.editor import TemplateAdopter
from src.dbbuilder import TextParser, TextChunker, VectorBuilder, GraphBuilder
from src.retriever import QueryProcessor, VectorDBRetrievalProcessor, GraphDBRetrievalProcessor, InfoRetriever
from src.planner import TaskPlanner
from src.executor import ContextManager, ExecutionEngine
from src.publisher import ConfluencePublisher

logger = get_logger(__name__)


class AutoLMRunner:
    """
    Main orchestrator for the AutoLM system.
    Implements the complete data flow pipeline from goal to task completion.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the AutoLM Runner with all necessary components.
        
        Args:
            config_path: Path to main configuration file
        """
        logger.info("Initializing AutoLM Runner")
        start_time = time.time()
        
        # Load configuration
        if config_path is None:
            config_path = Path.cwd() / "config" / "main_config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize utilities
        self.data_utility = DataUtility()
        self.ai_utility = AIUtility()
        self.stats_utility = StatisticsUtility()
        
        # Initialize core components
        self.generator = Generator()
        self.meta_generator = MetaGenerator(generator=self.generator)
        self.evaluator = Evaluator(generator=self.generator)
        self.template_adopter = TemplateAdopter()
        
        # Initialize database builders
        self.text_parser = TextParser()
        self.text_chunker = TextChunker()
        self.vector_builder = VectorBuilder(
            parser=self.text_parser,
            chunker=self.text_chunker,
            generator=self.generator
        )
        self.graph_builder = None  # Will be initialized after vector DB creation
        
        # Initialize context management and execution
        self.context_manager = ContextManager(config=self.config)
        self.execution_engine = ExecutionEngine(
            config=self.config,
            context_manager=self.context_manager
        )
        
        # Initialize task planner
        self.task_planner = TaskPlanner(
            config_file_path=config_path,
            generator=self.generator
        )
        
        # Initialize publisher (will be configured if Confluence details provided)
        self.publisher = None
        
        # Runtime state
        self.knowledge_base_established = False
        self.vector_db_path = None
        self.graph_db_path = None
        self.current_goal = None
        self.current_plan = None
        self.execution_results = []
        
        logger.info(f"AutoLM Runner initialized in {time.time() - start_time:.2f} seconds")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file."""
        try:
            if self.config_path.exists():
                config = self.data_utility.text_operation('load', self.config_path, file_type='json')
                logger.debug(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            "system": {
                "api": {"max_attempts": 3, "wait_time": 60},
                "paths": {
                    "knowledge_base": "db",
                    "source": "data"
                },
                "supported_file_types": ["pdf", "md", "txt"]
            },
            "knowledge_base": {
                "chunking": {
                    "method": "hierarchy",
                    "other_config": {
                        "chunk_size": 1000,
                        "chunk_overlap": 100
                    }
                },
                "vector_store": {
                    "embedding_dim": 1536,
                    "similarity_threshold": 0.7,
                    "top_k": 5
                }
            }
        }

    def save_knowledge_state(self):
        """Save all knowledge and memory to persistent storage."""
        # Persist memory to parquet files
        self.context_manager.persist_memory()
        
        # Ensure vector DB is saved
        if hasattr(self, 'vector_db') and self.vector_db is not None:
            os.makedirs("db/vector", exist_ok=True)
            self.vector_db.to_parquet("db/vector/main_vector_db.parquet")
        
        # Ensure graph DB is saved
        if hasattr(self, 'graph_db') and self.graph_db is not None:
            os.makedirs("db/graph", exist_ok=True)
            pd.to_pickle(self.graph_db, "db/graph/main_graph_db.pkl")

    def establish_knowledge_base(self, 
                                source_files: Optional[List[Union[str, Path]]] = None,
                                df_headings: Optional[Any] = None,
                                force_rebuild: bool = False) -> Tuple[str, str]:
        """
        Step 1: Establish knowledge base from source documents.
        
        Args:
            source_files: List of source files to process
            df_headings: Optional DataFrame with heading metadata
            force_rebuild: Whether to force rebuilding even if KB exists
            
        Returns:
            Tuple of (vector_db_path, graph_db_path)
        """
        logger.info("=== Step 1: Knowledge Base Establishment ===")
        start_time = time.time()
        
        try:
            # Check if knowledge base already exists
            if not force_rebuild and self.knowledge_base_established:
                logger.info("Knowledge base already established")
                return self.vector_db_path, self.graph_db_path
            
            # Use context manager to establish knowledge base
            vector_db_path, graph_db_path = self.context_manager.establish_knowledge_base(
                source_files=source_files,
                df_headings=df_headings
            )
            
            # Store paths and update state
            self.vector_db_path = vector_db_path
            self.graph_db_path = graph_db_path
            self.knowledge_base_established = True
            
            # Initialize graph builder with the created paths
            self.graph_builder = GraphBuilder(vectordb_file=vector_db_path)
            
            establishment_time = time.time() - start_time
            logger.info(f"Knowledge base establishment completed in {establishment_time:.2f} seconds")
            logger.info(f"Vector DB: {vector_db_path}")
            logger.info(f"Graph DB: {graph_db_path}")
            
            return vector_db_path, graph_db_path
            
        except Exception as e:
            logger.error(f"Knowledge base establishment failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
    
    def plan_task_sequence(self, 
                          goal: str,
                          max_prompts: Optional[int] = None,
                          similarity_threshold: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 2: Plan task sequence based on the provided goal.
        
        Args:
            goal: User-provided goal description
            max_prompts: Maximum number of prompts to select
            similarity_threshold: Minimum similarity threshold for prompt selection
            
        Returns:
            Tuple of (prompt_flow_config, shortlisted_prompts)
        """
        logger.info("=== Step 2: Goal-driven Task Planning ===")
        logger.info(f"Planning for goal: '{goal[:100]}{'...' if len(goal) > 100 else ''}'")
        start_time = time.time()
        
        try:
            # Store current goal
            self.current_goal = goal
            
            # Analyze goal complexity
            complexity_analysis = self.task_planner.analyze_goal_complexity(goal)
            logger.info(f"Goal complexity: {complexity_analysis.get('complexity_level', 'unknown')}")
            
            # Plan task sequence
            prompt_flow_config, shortlisted_prompts = self.task_planner.plan_task_sequence(
                goal=goal,
                max_prompts=max_prompts,
                similarity_threshold=similarity_threshold
            )
            
            # Store current plan
            self.current_plan = {
                'prompt_flow_config': prompt_flow_config,
                'shortlisted_prompts': shortlisted_prompts,
                'complexity_analysis': complexity_analysis
            }
            
            # Save planning results
            output_dir = Path.cwd() / "output" / "planning"
            self.task_planner.save_planning_results(
                prompt_flow_config=prompt_flow_config,
                shortlisted_prompts=shortlisted_prompts,
                output_dir=output_dir
            )
            
            planning_time = time.time() - start_time
            logger.info(f"Task planning completed in {planning_time:.2f} seconds")
            logger.info(f"Selected {len(shortlisted_prompts)} task prompts")
            
            return prompt_flow_config, shortlisted_prompts
            
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
    
    def setup_context_retrieval(self, shortlisted_prompts: Dict[str, Any]) -> List[str]:
        """
        Step 3: Setup context retrieval and populate memory systems.
        
        Args:
            shortlisted_prompts: Dictionary of selected task prompts
            
        Returns:
            List of memory IDs for stored contexts
        """
        logger.info("=== Step 3: Context Retrieval and Memory Management ===")
        start_time = time.time()
        
        try:
            # Ensure knowledge base is established
            if not self.knowledge_base_established:
                raise ValueError("Knowledge base must be established before context retrieval")
            
            # Identify common contexts across tasks
            common_contexts = self.context_manager.identify_common_contexts(shortlisted_prompts)
            logger.info(f"Identified {len(common_contexts)} common contexts")
            
            # Retrieve and store common contexts in goal memory
            memory_ids = self.context_manager.retrieve_short_term_goal_memory(common_contexts)
            logger.info(f"Stored {len(memory_ids)} contexts in goal memory")
            
            retrieval_time = time.time() - start_time
            logger.info(f"Context retrieval completed in {retrieval_time:.2f} seconds")
            
            return memory_ids
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
    
    def execute_task_sequence(self, 
                             prompt_flow_config: Dict[str, Any],
                             shortlisted_prompts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Step 4: Execute task sequence with optimization and memory management.
        
        Args:
            prompt_flow_config: Workflow configuration
            shortlisted_prompts: Selected task prompts
            
        Returns:
            List of execution results
        """
        logger.info("=== Step 4: Task Execution with Optimization ===")
        start_time = time.time()
        
        try:
            # Convert prompt flow config to execution plan
            execution_plan = self._convert_to_execution_plan(
                prompt_flow_config, 
                shortlisted_prompts
            )
            
            # Execute tasks using orchestrated execution flow
            execution_results = self.execution_engine.orchestrate_execution_flow(
                plan=execution_plan,
                overall_goal_context=self.current_goal
            )
            
            # Store execution results
            self.execution_results = execution_results
            
            # Save execution results
            output_dir = Path.cwd() / "output" / "execution"
            self.data_utility.ensure_directory(output_dir)
            
            for i, result in enumerate(execution_results):
                result_file = output_dir / f"task_{i+1}_result.json"
                self.data_utility.text_operation(
                    'save', 
                    result_file, 
                    result, 
                    file_type='json',
                    indent=2
                )
            
            execution_time = time.time() - start_time
            logger.info(f"Task execution completed in {execution_time:.2f} seconds")
            logger.info(f"Executed {len(execution_results)} tasks")
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
    
    def _convert_to_execution_plan(self, 
                                  prompt_flow_config: Dict[str, Any],
                                  shortlisted_prompts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert prompt flow config to execution plan format.
        
        Args:
            prompt_flow_config: Workflow configuration
            shortlisted_prompts: Selected task prompts
            
        Returns:
            List of task definitions for execution
        """
        execution_plan = []
        
        # Extract nodes and their order from prompt flow config
        nodes = prompt_flow_config.get('nodes', {})
        edges = prompt_flow_config.get('edges', [])
        
        # Create a simple topological order (for now, just use node order)
        # In a more sophisticated implementation, this would respect edge dependencies
        ordered_nodes = sorted(nodes.items(), key=lambda x: int(x[0].split('_')[1]))
        
        for node_id, node_data in ordered_nodes:
            # Find corresponding prompt in shortlisted_prompts
            task_prompt_id = node_data.get('task_prompt_id')
            task_summary = node_data.get('task_summary', '')
            task_type = node_data.get('task_type', 'deduction')
            
            # Find the actual prompt data
            prompt_data = None
            for prompt_id, prompt_info in shortlisted_prompts.items():
                if prompt_info.get('promptid') == task_prompt_id:
                    prompt_data = prompt_info
                    break
            
            if prompt_data:
                # Create execution task definition
                task_definition = {
                    'task_id': f"task_{prompt_data.get('promptid', node_id)}",
                    'task_prompt': prompt_data,
                    'task_type': task_type,
                    'confidence_level': 'medium',  # Default confidence level
                    'task_summary': task_summary
                }
                execution_plan.append(task_definition)
            else:
                logger.warning(f"Could not find prompt data for task_prompt_id: {task_prompt_id}")
        
        return execution_plan
    
    def evaluate_and_collate_results(self, 
                                    execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 5: Evaluate execution results and collate outputs.
        
        Args:
            execution_results: List of task execution results
            
        Returns:
            Collated evaluation results
        """
        logger.info("=== Step 5: Evaluation and Output Collation ===")
        start_time = time.time()
        
        try:
            # Evaluate individual task results
            evaluations = []
            for i, result in enumerate(execution_results):
                if result.get('response'):
                    # Evaluate response quality
                    task_prompt = result.get('metadata', {}).get('task_prompt', '')
                    response = result.get('response', '')
                    
                    # Create a basic evaluation
                    evaluation = {
                        'task_id': result.get('task_id', f'task_{i+1}'),
                        'response_length': len(response),
                        'response_word_count': len(response.split()),
                        'execution_time': result.get('performance_metrics', {}).get('execution_time', 0),
                        'task_type': result.get('metadata', {}).get('task_type', 'unknown'),
                        'success': result.get('response') is not None
                    }
                    evaluations.append(evaluation)
            
            # Collate overall results
            collated_results = {
                'goal': self.current_goal,
                'total_tasks': len(execution_results),
                'successful_tasks': sum(1 for e in evaluations if e['success']),
                'total_execution_time': sum(e['execution_time'] for e in evaluations),
                'average_response_length': sum(e['response_length'] for e in evaluations) / len(evaluations) if evaluations else 0,
                'task_evaluations': evaluations,
                'raw_results': execution_results
            }
            
            # Save collated results
            output_dir = Path.cwd() / "output" / "final"
            self.data_utility.ensure_directory(output_dir)
            
            final_results_file = output_dir / "collated_results.json"
            self.data_utility.text_operation(
                'save',
                final_results_file,
                collated_results,
                file_type='json',
                indent=2
            )
            
            evaluation_time = time.time() - start_time
            logger.info(f"Evaluation and collation completed in {evaluation_time:.2f} seconds")
            logger.info(f"Success rate: {collated_results['successful_tasks']}/{collated_results['total_tasks']}")
            
            return collated_results
            
        except Exception as e:
            logger.error(f"Evaluation and collation failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
    
    def publish_results(self, 
                       collated_results: Dict[str, Any],
                       confluence_config: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Step 6: Publish results to external platforms (optional).
        
        Args:
            collated_results: Collated evaluation results
            confluence_config: Optional Confluence configuration
            
        Returns:
            URL or ID of published content (if successful)
        """
        logger.info("=== Step 6: Publishing Results ===")
        start_time = time.time()
        
        try:
            if confluence_config:
                # Initialize Confluence publisher
                self.publisher = ConfluencePublisher(
                    confluence_url=confluence_config['url'],
                    username=confluence_config['username'],
                    api_token=confluence_config['api_token']
                )
                
                # Create publication content
                page_title = f"AutoLM Results - {self.current_goal[:50]}..."
                content = self._create_confluence_content(collated_results)
                
                # Publish to Confluence
                response = self.publisher.publish_page(
                    page_title=page_title,
                    content=content,
                    space_key=confluence_config['space_key'],
                    parent_id=confluence_config.get('parent_id')
                )
                
                page_id = response.get('id')
                logger.info(f"Published results to Confluence: {page_id}")
                
                publishing_time = time.time() - start_time
                logger.info(f"Publishing completed in {publishing_time:.2f} seconds")
                
                return page_id
            else:
                logger.info("No Confluence configuration provided, skipping publishing")
                return None
                
        except Exception as e:
            logger.error(f"Publishing failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            return None
    
    def _create_confluence_content(self, collated_results: Dict[str, Any]) -> str:
        """
        Create Confluence-formatted content from collated results.
        
        Args:
            collated_results: Collated evaluation results
            
        Returns:
            HTML content for Confluence
        """
        # Create basic HTML content
        content = f"""
        <h1>AutoLM Execution Results</h1>
        
        <h2>Goal</h2>
        <p>{collated_results['goal']}</p>
        
        <h2>Summary</h2>
        <ul>
            <li>Total Tasks: {collated_results['total_tasks']}</li>
            <li>Successful Tasks: {collated_results['successful_tasks']}</li>
            <li>Success Rate: {collated_results['successful_tasks'] / collated_results['total_tasks'] * 100:.1f}%</li>
            <li>Total Execution Time: {collated_results['total_execution_time']:.2f} seconds</li>
            <li>Average Response Length: {collated_results['average_response_length']:.0f} characters</li>
        </ul>
        
        <h2>Task Results</h2>
        <table>
            <tr>
                <th>Task ID</th>
                <th>Task Type</th>
                <th>Success</th>
                <th>Response Length</th>
                <th>Execution Time</th>
            </tr>
        """
        
        for eval_result in collated_results['task_evaluations']:
            content += f"""
            <tr>
                <td>{eval_result['task_id']}</td>
                <td>{eval_result['task_type']}</td>
                <td>{'✓' if eval_result['success'] else '✗'}</td>
                <td>{eval_result['response_length']}</td>
                <td>{eval_result['execution_time']:.2f}s</td>
            </tr>
            """
        
        content += """
        </table>
        
        <h2>Detailed Results</h2>
        <p>Detailed results are available in the exported JSON files.</p>
        """
        
        return content
    
    def run_full_pipeline(self, 
                         goal: str,
                         source_files: Optional[List[Union[str, Path]]] = None,
                         df_headings: Optional[Any] = None,
                         confluence_config: Optional[Dict[str, str]] = None,
                         force_rebuild_kb: bool = False) -> Dict[str, Any]:
        """
        Run the complete AutoLM pipeline from start to finish.
        
        Args:
            goal: User-provided goal description
            source_files: Optional list of source files for knowledge base
            df_headings: Optional DataFrame with heading metadata
            confluence_config: Optional Confluence publishing configuration
            force_rebuild_kb: Whether to force rebuilding the knowledge base
            
        Returns:
            Final collated results
        """
        logger.info("=== STARTING AUTOLM FULL PIPELINE ===")
        logger.info(f"Goal: {goal}")
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Establish Knowledge Base
            self.establish_knowledge_base(
                source_files=source_files,
                df_headings=df_headings,
                force_rebuild=force_rebuild_kb
            )
            
            # Step 2: Plan Task Sequence
            prompt_flow_config, shortlisted_prompts = self.plan_task_sequence(goal)
            
            # Step 3: Setup Context Retrieval
            memory_ids = self.setup_context_retrieval(shortlisted_prompts)
            
            # Step 4: Execute Task Sequence
            execution_results = self.execute_task_sequence(
                prompt_flow_config, 
                shortlisted_prompts
            )
            
            # Step 5: Evaluate and Collate Results
            collated_results = self.evaluate_and_collate_results(execution_results)
            
            # Step 6: Publish Results (optional)
            publication_id = self.publish_results(collated_results, confluence_config)
            if publication_id:
                collated_results['publication_id'] = publication_id
            
            # Add pipeline metadata
            total_time = time.time() - pipeline_start_time
            collated_results['pipeline_metadata'] = {
                'total_pipeline_time': total_time,
                'knowledge_base_path': self.vector_db_path,
                'graph_db_path': self.graph_db_path,
                'memory_contexts': len(memory_ids),
                'pipeline_success': True
            }
            
            logger.info("=== AUTOLM PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total pipeline time: {total_time:.2f} seconds")
            
            return collated_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Return error results
            error_results = {
                'error': str(e),
                'goal': goal,
                'pipeline_metadata': {
                    'total_pipeline_time': time.time() - pipeline_start_time,
                    'pipeline_success': False
                }
            }
            
            return error_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and component health.
        
        Returns:
            System status information
        """
        status = {
            'knowledge_base_established': self.knowledge_base_established,
            'vector_db_path': self.vector_db_path,
            'graph_db_path': self.graph_db_path,
            'current_goal': self.current_goal,
            'tasks_planned': len(self.current_plan.get('shortlisted_prompts', {})) if self.current_plan else 0,
            'tasks_executed': len(self.execution_results),
            'memory_goal_entries': len(self.context_manager.get_goal_memory()),
            'memory_procedural_entries': len(self.context_manager.get_procedural_memory()),
            'config_loaded': bool(self.config),
            'components_initialized': {
                'generator': self.generator is not None,
                'evaluator': self.evaluator is not None,
                'task_planner': self.task_planner is not None,
                'context_manager': self.context_manager is not None,
                'execution_engine': self.execution_engine is not None,
                'publisher': self.publisher is not None
            }
        }
        
        return status


def main():
    """Main entry point for the AutoLM Runner."""
    parser = argparse.ArgumentParser(description='AutoLM System Runner')
    parser.add_argument(
        '--goal', 
        type=str, 
        required=True,
        help='Goal description for task planning'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/main_config.json',
        help='Path to main configuration file'
    )
    parser.add_argument(
        '--sources', 
        type=str, 
        nargs='+',
        help='Source files for knowledge base'
    )
    parser.add_argument(
        '--rebuild-kb', 
        action='store_true',
        help='Force rebuilding of knowledge base'
    )
    parser.add_argument(
        '--publish-confluence', 
        action='store_true',
        help='Publish results to Confluence (requires confluence_config.json)'
    )
    parser.add_argument(
        '--status-only', 
        action='store_true',
        help='Only show system status without running pipeline'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = AutoLMRunner(config_path=args.config)
        
        if args.status_only:
            # Show system status only
            status = runner.get_system_status()
            print(json.dumps(status, indent=2))
            return
        
        # Setup Confluence configuration if requested
        confluence_config = None
        if args.publish_confluence:
            confluence_config_path = Path('config/confluence_config.json')
            if confluence_config_path.exists():
                confluence_config = runner.data_utility.text_operation(
                    'load', 
                    confluence_config_path, 
                    file_type='json'
                )
            else:
                logger.warning("Confluence publishing requested but config file not found")
        
        # Run full pipeline
        results = runner.run_full_pipeline(
            goal=args.goal,
            source_files=args.sources,
            confluence_config=confluence_config,
            force_rebuild_kb=args.rebuild_kb
        )
        
        # Print summary
        if results.get('pipeline_metadata', {}).get('pipeline_success', False):
            print("\n=== PIPELINE SUMMARY ===")
            print(f"Goal: {results['goal']}")
            print(f"Success Rate: {results['successful_tasks']}/{results['total_tasks']}")
            print(f"Total Time: {results['pipeline_metadata']['total_pipeline_time']:.2f}s")
            if 'publication_id' in results:
                print(f"Published: {results['publication_id']}")
        else:
            print(f"\n=== PIPELINE FAILED ===")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())