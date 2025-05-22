# Source Code Structure Summary

This document outlines the hierarchical structure of modules, classes, and functions within the `src` directory.

## Mermaid Diagram Visualization

```mermaid
graph TD
    root["src/"]

    subgraph __random__py ["__random__.py"]
        direction LR
        L_random_empty["(No classes or functions)"]
    end
    root --> __random__py

    subgraph contextualiser_py ["contextualiser.py"]
        direction LR
        C_ContextManager["ContextManager"]
        C_ContextManager --> F_ContextManager_init["__init__"]
        C_ContextManager --> F_ContextManager_persist_memory["persist_memory"]
        C_ContextManager --> F_ContextManager_store_task_memory["store_task_memory"]
        C_ContextManager --> F_ContextManager_store_goal_memory["store_goal_memory"]
        C_ContextManager --> F_ContextManager_get_task_memory["get_task_memory"]
        C_ContextManager --> F_ContextManager_get_goal_memory["get_goal_memory"]
        C_ContextManager --> F_ContextManager_store_procedural_memory["store_procedural_memory"]
        C_ContextManager --> F_ContextManager_get_procedural_memory["get_procedural_memory"]
        C_ContextManager --> F_ContextManager_establish_knowledge_base["establish_knowledge_base"]
        C_ContextManager --> F_ContextManager_initialize_retrieval_processors["_initialize_retrieval_processors"]
        C_ContextManager --> F_ContextManager_discover_source_files["_discover_source_files"]
        C_ContextManager --> F_ContextManager_convert_pdf_to_markdown["_convert_pdf_to_markdown"]
        C_ContextManager --> F_ContextManager_create_vector_database["_create_vector_database"]
        C_ContextManager --> F_ContextManager_create_graph_database["_create_graph_database"]
        C_ContextManager --> F_ContextManager_identify_common_contexts["identify_common_contexts"]
        C_ContextManager --> F_ContextManager_extract_context_requirements["_extract_context_requirements"]
        C_ContextManager --> F_ContextManager_extract_keywords["_extract_keywords"]
        C_ContextManager --> F_ContextManager_find_common_contexts["_find_common_contexts"]
        C_ContextManager --> F_ContextManager_retrieve_short_term_goal_memory["retrieve_short_term_goal_memory"]
        C_ContextManager --> F_ContextManager_assess_context_sufficiency["assess_context_sufficiency"]
        C_ContextManager --> F_ContextManager_retrieve_task_specific_context["retrieve_task_specific_context"]
    end
    root --> contextualiser_py

    subgraph dbbuilder_py ["dbbuilder.py"]
        direction LR
        C_TextParser["TextParser"]
        C_TextParser --> F_TextParser_init["__init__"]
        C_TextParser --> F_TextParser_pdf2md_markitdown["pdf2md_markitdown"]
        C_TextParser --> F_TextParser_pdf2md_openleaf["pdf2md_openleaf"]
        C_TextParser --> F_TextParser_pdf2md_ocr["pdf2md_ocr"]
        C_TextParser --> F_TextParser_pdf2md_llamaindex["pdf2md_llamaindex"]
        C_TextParser --> F_TextParser_pdf2md_pymupdf["pdf2md_pymupdf"]
        C_TextParser --> F_TextParser_pdf2md_textract["pdf2md_textract"]

        C_TextChunker["TextChunker"]
        C_TextChunker --> F_TextChunker_init["__init__"]
        C_TextChunker --> F_TextChunker_length_based_chunking["length_based_chunking"]
        C_TextChunker --> F_TextChunker_hierarchy_based_chunking["hierarchy_based_chunking"]
        C_TextChunker --> F_TextChunker_append_section["_append_section"]

        C_VectorBuilder["VectorBuilder"]
        C_VectorBuilder --> F_VectorBuilder_init["__init__"]
        C_VectorBuilder --> F_VectorBuilder_create_vectordb["create_vectordb"]
        C_VectorBuilder --> F_VectorBuilder_load_vectordb["load_vectordb"]
        C_VectorBuilder --> F_VectorBuilder_merge_vectordbs["merge_vectordbs"]
        C_VectorBuilder --> F_VectorBuilder_extract_references_with_lm["_extract_references_with_lm"]
        C_VectorBuilder --> F_VectorBuilder_extract_reference["extract_reference"]

        C_GraphBuilder["GraphBuilder"]
        C_GraphBuilder --> F_GraphBuilder_init["__init__"]
        C_GraphBuilder --> F_GraphBuilder_lookup_node["lookup_node"]
        C_GraphBuilder --> F_GraphBuilder_list_nodes["list_nodes"]
        C_GraphBuilder --> F_GraphBuilder_build_graph["build_graph"]
        C_GraphBuilder --> F_GraphBuilder_find_similar_nodes["_find_similar_nodes"]
        C_GraphBuilder --> F_GraphBuilder_build_hypergraph["build_hypergraph"]
        C_GraphBuilder --> F_GraphBuilder_cluster_by_embedding["_cluster_by_embedding"]
        C_GraphBuilder --> F_GraphBuilder_find_reference_chains["_find_reference_chains"]
        F_GraphBuilder_find_reference_chains --> F_GraphBuilder_dfs["dfs (nested)"]
        C_GraphBuilder --> F_GraphBuilder_save_graphdb["save_graphdb"]
        C_GraphBuilder --> F_GraphBuilder_load_graphdb["load_graphdb"]
        C_GraphBuilder --> F_GraphBuilder_merge_graphdbs["merge_graphdbs"]
        C_GraphBuilder --> F_GraphBuilder_view_graphdb["view_graphdb"]
    end
    root --> dbbuilder_py

    subgraph editor_py ["editor.py"]
        direction LR
        C_TemplateAdopter["TemplateAdopter"]
        C_TemplateAdopter --> F_TemplateAdopter_init["__init__"]
        C_TemplateAdopter --> F_TemplateAdopter_get_prompt_transformation["get_prompt_transformation"]
    end
    root --> editor_py

    subgraph evaluator_py ["evaluator.py"]
        direction LR
        C_Evaluator["Evaluator"]
        C_Evaluator --> F_Evaluator_init["__init__"]
        C_Evaluator --> F_Evaluator_get_tokenisation["get_tokenisation"]
        C_Evaluator --> F_Evaluator_generate_ngrams["_generate_ngrams"]
        C_Evaluator --> F_Evaluator_get_bleu["get_bleu"]
        C_Evaluator --> F_Evaluator_get_rouge["get_rouge"]
        C_Evaluator --> F_Evaluator_get_bertscore["get_bertscore"]
        C_Evaluator --> F_Evaluator_get_meteor["get_meteor"]
        C_Evaluator --> F_Evaluator_get_chunks["_get_chunks"]
        C_Evaluator --> F_Evaluator_evaluate_all["evaluate_all"]

        C_PlanEvaluator["PlanEvaluator"]
        C_PlanEvaluator --> F_PlanEvaluator_init["__init__"]
        C_PlanEvaluator --> F_PlanEvaluator_evaluate_plan["evaluate_plan"]

        C_RetrievalEvaluator["RetrievalEvaluator"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_init["__init__"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_evaluate_retrieval_quality["evaluate_retrieval_quality"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_evaluate_query_performance["evaluate_query_performance"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_calculate_rank_metrics["_calculate_rank_metrics"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_calculate_ndcg["_calculate_ndcg"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_calculate_map["_calculate_map"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_analyze_term_overlap["_analyze_term_overlap"]
        C_RetrievalEvaluator --> F_RetrievalEvaluator_estimate_query_effectiveness["_estimate_query_effectiveness"]
    end
    root --> evaluator_py

    subgraph executor_py ["executor.py"]
        direction LR
        C_ExecutionEngine["ExecutionEngine"]
        C_ExecutionEngine --> F_ExecutionEngine_init["__init__"]
        C_ExecutionEngine --> F_ExecutionEngine_load_mle_config["_load_mle_config"]
        C_ExecutionEngine --> F_ExecutionEngine_execute_task["execute_task"]
        C_ExecutionEngine --> F_ExecutionEngine_determine_task_type_and_confidence["_determine_task_type_and_confidence"]
        C_ExecutionEngine --> F_ExecutionEngine_heuristic_task_analysis["_heuristic_task_analysis"]
        C_ExecutionEngine --> F_ExecutionEngine_get_execution_config["_get_execution_config"]
        C_ExecutionEngine --> F_ExecutionEngine_prepare_context["_prepare_context"]
        C_ExecutionEngine --> F_ExecutionEngine_apply_prompt_optimization["_apply_prompt_optimization"]
        C_ExecutionEngine --> F_ExecutionEngine_execute_with_topology["_execute_with_topology"]
        C_ExecutionEngine --> F_ExecutionEngine_update_task_iteration_count["_update_task_iteration_count"]
        C_ExecutionEngine --> F_ExecutionEngine_prepare_contextual_prompt["_prepare_contextual_prompt"]
        C_ExecutionEngine --> F_ExecutionEngine_evaluate_response["_evaluate_response"]
        C_ExecutionEngine --> F_ExecutionEngine_handle_execution_fallback["_handle_execution_fallback"]
        C_ExecutionEngine --> F_ExecutionEngine_orchestrate_execution_flow["orchestrate_execution_flow"]
    end
    root --> executor_py

    subgraph generator_py ["generator.py"]
        direction LR
        C_Generator["Generator"]
        C_Generator --> F_Generator_init["__init__"]
        C_Generator --> F_Generator_validate_model["_validate_model"]
        C_Generator --> F_Generator_validate_parameters["_validate_parameters"]
        C_Generator --> F_Generator_refresh_token["refresh_token"]
        C_Generator --> F_Generator_get_completion["get_completion"]
        C_Generator --> F_Generator_get_azure_completion["_get_azure_completion"]
        C_Generator --> F_Generator_get_gemini_completion["_get_gemini_completion"]
        C_Generator --> F_Generator_get_claude_completion["_get_claude_completion"]
        C_Generator --> F_Generator_get_hf_completion["_get_hf_completion"]
        C_Generator --> F_Generator_get_embeddings["get_embeddings"]
        C_Generator --> F_Generator_get_azure_embeddings["_get_azure_embeddings"]
        C_Generator --> F_Generator_get_vertex_embeddings["_get_vertex_embeddings"]
        C_Generator --> F_Generator_get_anthropic_embeddings["_get_anthropic_embeddings"]
        C_Generator --> F_Generator_get_hf_embeddings["_get_hf_embeddings"]
        C_Generator --> F_Generator_get_reranking["get_reranking"]
        C_Generator --> F_Generator_get_reasoning_simulator["get_reasoning_simulator"]
        C_Generator --> F_Generator_get_azure_reasoning_simulator["_get_azure_reasoning_simulator"]
        C_Generator --> F_Generator_get_hf_reasoning_simulator["_get_hf_reasoning_simulator"]
        C_Generator --> F_Generator_get_hf_ocr["get_hf_ocr"]
        C_Generator --> F_Generator_calculate_perplexity["_calculate_perplexity"]
        C_Generator --> F_Generator_get_voice["get_voice"]

        C_MetaGenerator["MetaGenerator"]
        C_MetaGenerator --> F_MetaGenerator_init["__init__"]
        C_MetaGenerator --> F_MetaGenerator_get_meta_generation["get_meta_generation"]

        C_Encoder["Encoder"]
        C_Encoder --> F_Encoder_init["__init__"]
        C_Encoder --> F_Encoder_load_model["_load_model"]
        C_Encoder --> F_Encoder_validate_model["_validate_model"]
        C_Encoder --> F_Encoder_encode["encode"]
        C_Encoder --> F_Encoder_predict["predict"]
        C_Encoder --> F_Encoder_rerank["rerank"]
    end
    root --> generator_py

    subgraph logging_py ["logging.py"]
        direction LR
        F_get_logger["get_logger"]
    end
    root --> logging_py

    subgraph planner_py ["planner.py"]
        direction LR
        C_TaskPlanner["TaskPlanner"]
        C_TaskPlanner --> F_TaskPlanner_init["__init__"]
        C_TaskPlanner --> F_TaskPlanner_plan_task_sequence["plan_task_sequence"]
        
        F_optimise_sequence["optimise_sequence"]
        F_incorporate_user_feedback["incorporate_user_feedback"]

        C_TaskAssessor["TaskAssessor"]
        C_TaskAssessor --> F_TaskAssessor_init["__init__"]
        C_TaskAssessor --> F_TaskAssessor_predict_task_type["predict_task_type"]
        C_TaskAssessor --> F_TaskAssessor_predict_task_complexity["predict_task_complexity"]
    end
    root --> planner_py

    subgraph publisher_py ["publisher.py"]
        direction LR
        C_ConfluencePublisher["ConfluencePublisher"]
        C_ConfluencePublisher --> F_ConfluencePublisher_init["__init__"]
        C_ConfluencePublisher --> F_ConfluencePublisher_convert_latex_to_confluence_math["convert_latex_to_confluence_math"]
        C_ConfluencePublisher --> F_ConfluencePublisher_markdown_to_html["markdown_to_html"]
        C_ConfluencePublisher --> F_ConfluencePublisher_generate_section_html["generate_section_html"]
        C_ConfluencePublisher --> F_ConfluencePublisher_build_full_page_html["build_full_page_html"]
        C_ConfluencePublisher --> F_ConfluencePublisher_publish_page["publish_page"]
    end
    root --> publisher_py

    subgraph retriever_py ["retriever.py"]
        direction LR
        C_QueryProcessor["QueryProcessor"]
        C_QueryProcessor --> F_QueryProcessor_init["__init__"]
        C_QueryProcessor --> F_QueryProcessor_rephrase_query["rephrase_query"]
        C_QueryProcessor --> F_QueryProcessor_decompose_query["decompose_query"]
        C_QueryProcessor --> F_QueryProcessor_hypothesize_query["hypothesize_query"]
        C_QueryProcessor --> F_QueryProcessor_predict_query["predict_query"]

        C_RerankProcessor["RerankProcessor"]
        C_RerankProcessor --> F_RerankProcessor_init["__init__"]
        C_RerankProcessor --> F_RerankProcessor_rerank_reciprocal_rank_fusion["rerank_reciprocal_rank_fusion"]
        C_RerankProcessor --> F_RerankProcessor_rerank_cross_encoder["rerank_cross_encoder"]

        C_VectorDBRetrievalProcessor["VectorDBRetrievalProcessor"]
        C_VectorDBRetrievalProcessor --> F_VectorDBRetrievalProcessor_init["__init__"]
        C_VectorDBRetrievalProcessor --> F_VectorDBRetrievalProcessor_load_embedding_model["_load_embedding_model"]
        C_VectorDBRetrievalProcessor --> F_VectorDBRetrievalProcessor_semantic_search["semantic_search"]
        C_VectorDBRetrievalProcessor --> F_VectorDBRetrievalProcessor_symbolic_search["symbolic_search"]
        C_VectorDBRetrievalProcessor --> F_VectorDBRetrievalProcessor_graph_based_search["graph_based_search"]

        C_GraphDBRetrievalProcessor["GraphDBRetrievalProcessor"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_init["__init__"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_semantic_cluster_search["semantic_cluster_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_concept_hierarchy_search["concept_hierarchy_search"]
        F_GraphDBRetrievalProcessor_concept_hierarchy_search --> F_GraphDBRetrievalProcessor_traverse_concepts["traverse_concepts (nested)"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_temporal_search["temporal_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_cross_layer_search["cross_layer_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_content_layer_search["_content_layer_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_structure_layer_search["_structure_layer_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_reference_layer_search["_reference_layer_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_attribute_layer_search["_attribute_layer_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_path_based_search["path_based_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_connectivity_search["connectivity_search"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_community_detection["community_detection"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_hypergraph_query["hypergraph_query"]
        F_GraphDBRetrievalProcessor_hypergraph_query --> F_GraphDBRetrievalProcessor_traverse_diffusion["traverse_diffusion (nested)"]
        C_GraphDBRetrievalProcessor --> F_GraphDBRetrievalProcessor_multilayer_hypergraph_fusion["multilayer_hypergraph_fusion"]

        C_InfoRetriever["InfoRetriever"]
        C_InfoRetriever --> F_InfoRetriever_init["__init__"]
        C_InfoRetriever --> F_InfoRetriever_vector_retrieval["vector_retrieval"]
        C_InfoRetriever --> F_InfoRetriever_graph_retrieval["graph_retrieval"]
        C_InfoRetriever --> F_InfoRetriever_hybrid_retrieval["hybrid_retrieval"]
    end
    root --> retriever_py

    subgraph topologist_py ["topologist.py"]
        direction LR
        C_PromptTopology["PromptTopology"]
        C_PromptTopology --> F_PromptTopology_init["__init__"]
        C_PromptTopology --> F_PromptTopology_prompt_disambiguation["prompt_disambiguation"]
        C_PromptTopology --> F_PromptTopology_prompt_genetic_algorithm["prompt_genetic_algorithm"]
        C_PromptTopology --> F_PromptTopology_prompt_differential["prompt_differential"]
        C_PromptTopology --> F_PromptTopology_prompt_breeder["prompt_breeder"]
        C_PromptTopology --> F_PromptTopology_prompt_phrase_evolution["prompt_phrase_evolution"]
        C_PromptTopology --> F_PromptTopology_prompt_persona_search["prompt_persona_search"]
        C_PromptTopology --> F_PromptTopology_prompt_examplar["prompt_examplar"]
        C_PromptTopology --> F_PromptTopology_prompt_reasoning["prompt_reasoning"]

        C_ScalingTopology["ScalingTopology"]
        C_ScalingTopology --> F_ScalingTopology_init["__init__"]
        C_ScalingTopology --> F_ScalingTopology_best_of_n_synthesis["best_of_n_synthesis"]
        C_ScalingTopology --> F_ScalingTopology_best_of_n_selection["best_of_n_selection"]
        C_ScalingTopology --> F_ScalingTopology_self_reflection["self_reflection"]
        C_ScalingTopology --> F_ScalingTopology_atom_of_thought["atom_of_thought"]
        C_ScalingTopology --> F_ScalingTopology_multimodel_debate_solo["multimodel_debate_solo"]
        C_ScalingTopology --> F_ScalingTopology_multimodel_debate_dual["multimodel_debate_dual"]
        C_ScalingTopology --> F_ScalingTopology_multipath_disambiguation_selection["multipath_disambiguation_selection"]
        C_ScalingTopology --> F_ScalingTopology_socratic_dialogue["socratic_dialogue"]
        C_ScalingTopology --> F_ScalingTopology_hierarchical_decomposition["hierarchical_decomposition"]
        F_ScalingTopology_hierarchical_decomposition --> F_ScalingTopology_topological_sort_hd["topological_sort (nested)"]
        F_ScalingTopology_topological_sort_hd --> F_ScalingTopology_visit_hd["visit (nested)"]
        C_ScalingTopology --> F_ScalingTopology_regenerative_majority_synthesis["regenerative_majority_synthesis"]
        C_ScalingTopology --> F_ScalingTopology_adaptive_dag_reasoning["adaptive_dag_reasoning"]
        F_ScalingTopology_adaptive_dag_reasoning --> F_ScalingTopology_process_hierarchical_node_adr["process_hierarchical_node (nested)"]
        F_ScalingTopology_adaptive_dag_reasoning --> F_ScalingTopology_topological_sort_adr["topological_sort (nested)"]
        F_ScalingTopology_adaptive_dag_reasoning --> F_ScalingTopology_add_hierarchical_solutions_adr["add_hierarchical_solutions (nested)"]
        C_ScalingTopology --> F_ScalingTopology_recursive_chain_of_thought["recursive_chain_of_thought"]
        C_ScalingTopology --> F_ScalingTopology_ensemble_weighted_voting["ensemble_weighted_voting"]
        C_ScalingTopology --> F_ScalingTopology_calculate_consistency_score["_calculate_consistency_score"]
        C_ScalingTopology --> F_ScalingTopology_jaccard_similarity["_jaccard_similarity"]
        C_ScalingTopology --> F_ScalingTopology_weighted_selection["_weighted_selection"]
    end
    root --> topologist_py

    subgraph utility_py ["utility.py"]
        direction LR
        C_DataUtility["DataUtility"]
        C_DataUtility --> F_DataUtility_ensure_directory["ensure_directory"]
        C_DataUtility --> F_DataUtility_text_operation["text_operation"]
        C_DataUtility --> F_DataUtility_csv_operation["csv_operation"]
        C_DataUtility --> F_DataUtility_format_conversion["format_conversion"]
        C_DataUtility --> F_DataUtility_dataframe_operations["dataframe_operations"]

        C_StatisticsUtility["StatisticsUtility"]
        C_StatisticsUtility --> F_StatisticsUtility_set_random_seed["set_random_seed"]

        C_AIUtility["AIUtility"]
        C_AIUtility --> F_AIUtility_load_prompts["_load_prompts"]
        C_AIUtility --> F_AIUtility_get_meta_prompt["get_meta_prompt"]
        C_AIUtility --> F_AIUtility_apply_meta_prompt["apply_meta_prompt"]
        C_AIUtility --> F_AIUtility_get_meta_fix["get_meta_fix"]
        C_AIUtility --> F_AIUtility_get_task_prompt["get_task_prompt"]
        C_AIUtility --> F_AIUtility_list_prompt_categories["list_prompt_categories"]
        F_AIUtility_list_prompt_categories --> F_AIUtility_collect_categories["collect_categories (nested)"]
        C_AIUtility --> F_AIUtility_list_task_prompts["list_task_prompts"]
        C_AIUtility --> F_AIUtility_get_encoder["_get_encoder"]
        C_AIUtility --> F_AIUtility_process_tokens["process_tokens"]
        C_AIUtility --> F_AIUtility_format_prompt_components["format_prompt_components"]
        C_AIUtility --> F_AIUtility_format_text_list["format_text_list"]
        C_AIUtility --> F_AIUtility_format_json_response["format_json_response"]
        C_AIUtility --> F_AIUtility_get_prompt_core_components["get_prompt_core_components"]
        C_AIUtility --> F_AIUtility_merge_response_format["merge_response_format"]
    end
    root --> utility_py
```

## Detailed Breakdown

### `__random__.py`
- (No classes or functions found)

### `contextualiser.py`
- **Class: ContextManager**
  - `__init__(self, config: Dict[str, Any])`
  - `persist_memory(self)`
  - `store_task_memory(self, task_id: str, memory_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str`
  - `store_goal_memory(self, entity_name: str, goal_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str`
  - `get_task_memory(self, task_id: str) -> List[Dict[str, Any]]`
  - `get_goal_memory(self, entity_name: Optional[str] = None) -> List[Dict[str, Any]]`
  - `store_procedural_memory(self, config_name: str, config_type: str, config_content: Dict[str, Any]) -> str`
  - `get_procedural_memory(self, config_type: Optional[str] = None) -> List[Dict[str, Any]]`
  - `establish_knowledge_base(self, source_type: str, source_path: Optional[str] = None, ...)`
  - `_initialize_retrieval_processors(self)`
  - `_discover_source_files(self) -> List[Path]`
  - `_convert_pdf_to_markdown(self, pdf_path: Path) -> Path`
  - `_create_vector_database(self, markdown_file: Path, df_headings: Optional[pd.DataFrame] = None) -> str`
  - `_create_graph_database(self, vector_db_path: str) -> str`
  - `identify_common_contexts(self, shortlisted_prompts: Dict[str, Any]) -> List[Dict[str, Any]]`
  - `_extract_context_requirements(self, prompt_id: str, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]`
  - `_extract_keywords(self, text: str) -> List[str]`
  - `_find_common_contexts(self, context_requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
  - `retrieve_short_term_goal_memory(self, common_contexts: List[Dict[str, Any]]) -> List[str]`
  - `assess_context_sufficiency(self, task_prompt: Dict[str, Any], task_id: str) -> Tuple[bool, Optional[str]]`
  - `retrieve_task_specific_context(self, task_id: str, query: str) -> str`

### `dbbuilder.py`
- **Class: TextParser**
  - `__init__(self) -> None`
  - `pdf2md_markitdown(self, pdf_path: str) -> None`
  - `pdf2md_openleaf(self, pdf_path: str) -> None`
  - `pdf2md_ocr(self, pdf_path: str, md_path: str, model: str = "GOT-OCR2") -> None`
  - `pdf2md_llamaindex(self, pdf_path: str) -> None`
  - `pdf2md_pymupdf(self, pdf_path: str) -> None`
  - `pdf2md_textract(self, pdf_path: str) -> None`
- **Class: TextChunker**
  - `__init__(self, config_file_path: Optional[Union[str, Path]] = None) -> None`
  - `length_based_chunking(self, markdown_file: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> pd.DataFrame`
  - `hierarchy_based_chunking(self, markdown_file: str, df_headings: pd.DataFrame) -> pd.DataFrame`
  - `_append_section(self, current_section: Dict[str, Any], sections: List[Dict[str, Any]], min_length: int)`
- **Class: VectorBuilder**
  - `__init__(self, parser: TextParser, chunker: TextChunker, generator=None, ...)`
  - `create_vectordb(self, markdown_file: str, ...)`
  - `load_vectordb(self, parquet_file: str = None) -> pd.DataFrame`
  - `merge_vectordbs(self, parquet_files: List[str], output_name: str = None) -> pd.DataFrame`
  - `_extract_references_with_lm(self, text: str, generator: Any) -> List[str]`
  - `extract_reference(self, vectordb_file: str) -> pd.DataFrame`
- **Class: GraphBuilder**
  - `__init__(self, vectordb_file: str = None, db_path: Optional[Union[str, Path]] = None) -> None`
  - `lookup_node(self, node_reference: str) -> dict`
  - `list_nodes(self) -> list`
  - `build_graph(self, enhanced_df: Optional[pd.DataFrame] = None) -> nx.MultiDiGraph`
  - `_find_similar_nodes(self, row: pd.Series, df: pd.DataFrame, threshold: float = 0.8)`
  - `build_hypergraph(self, enhanced_df: Optional[pd.DataFrame] = None) -> nx.Graph`
  - `_cluster_by_embedding(self, df: pd.DataFrame, n_clusters: int = 10) -> List[List[str]]`
  - `_find_reference_chains(self, df: pd.DataFrame) -> List[List[str]]`
    - `dfs(node: str, current_chain: List[str])` (nested function)
  - `save_graphdb(self, graph_type: str = 'standard', custom_name: str = None) -> str`
  - `load_graphdb(self, filepath: str = None, graph_type: str = 'standard') -> nx.Graph`
  - `merge_graphdbs(self, filepaths: List[str], output_name: str = None, graph_type: str = 'standard') -> nx.Graph`
  - `view_graphdb(self, graph_type: str = 'standard', output_path: str = None, ...)`

### `editor.py`
- **Class: TemplateAdopter**
  - `__init__(self)`
  - `get_prompt_transformation(self, template_type: str, template_name: str, parameters: Dict[str, Any]) -> str`

### `evaluator.py`
- **Class: Evaluator**
  - `__init__(self, generator=None)`
  - `get_tokenisation(self, text: str, granularity: str = 'word') -> List[str]`
  - `_generate_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]`
  - `get_bleu(self, reference_text: str, generated_text: str, n: int = 4, ...)`
  - `get_rouge(self, reference_text: str, generated_text: str, rouge_type: str = 'rouge-l', ...)`
  - `get_bertscore(self, reference_text: str, generated_text: str, model_type: str = 'bert-base-uncased', ...)`
  - `get_meteor(self, reference_text: str, generated_text: str, alpha: float = 0.9, ...)`
  - `_get_chunks(self, gen_tokens: List[str], ref_tokens: List[str]) -> List[List[str]]`
  - `evaluate_all(self, reference_text: str, generated_text: str, ...)`
- **Class: PlanEvaluator**
  - `__init__(self, generator: Any = None, evaluator: Evaluator = None)`
  - `evaluate_plan(self, plan: str) -> Dict[str, float]`
- **Class: RetrievalEvaluator**
  - `__init__(self, generator: Any = None, evaluator: Evaluator = None)`
  - `evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Dict[str, Any]], ...)`
  - `evaluate_query_performance(self, original_query: str, processed_query: str, ...)`
  - `_calculate_rank_metrics(self, retrieved_docs: List[Dict[str, Any]], ground_truth_docs: List[Dict[str, Any]], k: int) -> Dict[str, float]`
  - `_calculate_ndcg(self, relevance: List[int], k: int = 10) -> float`
  - `_calculate_map(self, relevance: List[int]) -> float`
  - `_analyze_term_overlap(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]`
  - `_estimate_query_effectiveness(self, term_overlap: Dict[str, float]) -> float`

### `executor.py`
- **Class: ExecutionEngine**
  - `__init__(self, config: Dict[str, Any], context_manager: ContextManager)`
  - `_load_mle_config(self) -> Dict[str, Any]`
  - `execute_task(self, task_id: str, task_prompt: Dict[str, Any], ...)`
  - `_determine_task_type_and_confidence(self, task_prompt: Dict[str, Any]) -> Tuple[str, str]`
  - `_heuristic_task_analysis(self, analysis_text: str) -> Tuple[str, str]`
  - `_get_execution_config(self, task_type: str, confidence_level: str) -> Dict[str, Any]`
  - `_prepare_context(self, task_id: str, task_prompt: Dict[str, Any]) -> Dict[str, Any]`
  - `_apply_prompt_optimization(self, task_prompt: Dict[str, Any], execution_config: Dict[str, Any]) -> Dict[str, Any]`
  - `_execute_with_topology(self, task_id: str, contextual_prompt: str, ...)`
  - `_update_task_iteration_count(self, task_id: str, task_prompt: Dict[str, Any]) -> None`
  - `_prepare_contextual_prompt(self, task_prompt: Dict[str, Any], context_data: Dict[str, Any]) -> str`
  - `_evaluate_response(self, prompt: Dict[str, Any], response: str) -> Dict[str, Any]`
  - `_handle_execution_fallback(self, task_id: str, task_prompt: Dict[str, Any]) -> Dict[str, Any]`
  - `orchestrate_execution_flow(self, plan: List[Dict[str, Any]], overall_goal_context: Optional[str] = None) -> List[Dict[str, Any]]`

### `generator.py`
- **Class: Generator**
  - `__init__(self)`
  - `_validate_model(self, model_name: str, model_type: str, provider: str = None) -> bool`
  - `_validate_parameters(self, params: Dict[str, Any], model_type: str) -> bool`
  - `refresh_token(self) -> str`
  - `get_completion(self, prompt: str, model_name: str, model_type: str, ...)`
  - `_get_azure_completion(self, prompt: str, model_name: str, params: Dict[str, Any], ...)`
  - `_get_gemini_completion(self, prompt: str, model_name: str, params: Dict[str, Any], ...)`
  - `_get_claude_completion(self, prompt: str, model_name: str, params: Dict[str, Any], ...)`
  - `_get_hf_completion(self, prompt: str, model_name: str, params: Dict[str, Any], ...)`
  - `get_embeddings(self, text: Union[str, List[str]], model_name: str, model_type: str, ...)`
  - `_get_azure_embeddings(self, text: Union[str, List[str]], model: str, ...)`
  - `_get_vertex_embeddings(self, text: Union[str, List[str]], model_name: str, params: Dict[str, Any], ...)`
  - `_get_anthropic_embeddings(self, text: Union[str, List[str]], model_name: str, params: Dict[str, Any], ...)`
  - `_get_hf_embeddings(self, text: Union[str, List[str]], model_name: str, params: Dict[str, Any], ...)`
  - `get_reranking(self, query: str, documents: List[str], model_name: str, model_type: str, ...)`
  - `get_reasoning_simulator(self, prompt: str, model_name: str, model_type: str, ...)`
  - `_get_azure_reasoning_simulator(self, prompt: str, model_name: str, params: Dict[str, Any], ...)`
  - `_get_hf_reasoning_simulator(self, prompt: str, model_name: str, params: Dict[str, Any], ...)`
  - `get_hf_ocr(self, image_path: str, model_name: str = "microsoft/trocr-base-handwritten", ...)`
  - `_calculate_perplexity(self, logprobs: Dict[str, Any]) -> float`
  - `get_voice(self)`
- **Class: MetaGenerator**
  - `__init__(self, generator=None)`
  - `get_meta_generation(self, base_prompt: str, meta_prompt_template: str, ...)`
- **Class: Encoder**
  - `__init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", ...)`
  - `_load_model(self)`
  - `_validate_model(self, model_name: str, model_type: str) -> bool`
  - `encode(self, text: Union[str, List[str]], **kwargs) -> np.ndarray`
  - `predict(self, texts: Union[List[str], List[Tuple[str, str]]], **kwargs) -> np.ndarray`
  - `rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]`

### `logging.py`
- `get_logger(name: str)`

### `planner.py`
- **Class: TaskPlanner**
  - `__init__(self, generator: Generator, context_manager: ContextManager, evaluator: PlanEvaluator, ...)`
  - `plan_task_sequence(self, overall_goal: str, ...)`
- `optimise_sequence()`
- `incorporate_user_feedback()`
- **Class: TaskAssessor**
  - `__init__(self, config_file_path: Optional[Union[str, Path]] = None)`
  - `predict_task_type(self, task_prompt: str) -> str`
  - `predict_task_complexity(self, task_prompt: str) -> str`

### `publisher.py`
- **Class: ConfluencePublisher**
  - `__init__(self, confluence_url: str, username: str, api_token: str, cloud: bool = True)`
  - `convert_latex_to_confluence_math(text: str) -> str`
  - `markdown_to_html(text: str) -> str`
  - `generate_section_html(self, title: str, content: str, level: int = 1, is_open: bool = True) -> str`
  - `build_full_page_html(self, title: str, sections: List[Dict[str, Any]], toc: bool = True) -> str`
  - `publish_page(self, space_key: str, title: str, body: str, ...)`

### `retriever.py`
- **Class: QueryProcessor**
  - `__init__(self, generator: Optional[Generator] = None)`
  - `rephrase_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5) -> str`
  - `decompose_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.7) -> List[Dict[str, Any]]`
  - `hypothesize_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5) -> str`
  - `predict_query(self, query: str, response: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5)`
- **Class: RerankProcessor**
  - `__init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "mps")`
  - `rerank_reciprocal_rank_fusion(self, results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]`
  - `rerank_cross_encoder(self, query: str, results: List[Dict[str, Any]])`
- **Class: VectorDBRetrievalProcessor**
  - `__init__(self, vector_builder=None, vector_file=None, generator=None)`
  - `_load_embedding_model(self, model_name: Optional[str] = None) -> SentenceTransformer`
  - `semantic_search(self, query: Union[str, np.ndarray], corpus: List[Dict[str, Any]], ...)`
  - `symbolic_search(self, query_terms: List[str], corpus: List[Dict[str, Any]], ...)`
  - `graph_based_search(self, query: str, graph: nx.Graph, top_k: int = 5, search_depth: int = 2) -> List[Dict[str, Any]]`
- **Class: GraphDBRetrievalProcessor**
  - `__init__(self, graph_builder=None, graph_file=None, graph_type='standard')`
  - `semantic_cluster_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]`
  - `concept_hierarchy_search(self, query: str, max_depth: int = 3) -> List[Dict[str, Any]]`
    - `traverse_concepts(edge, depth=0)` (nested function)
  - `temporal_search(self, start_date: str, end_date: str = None) -> List[Dict[str, Any]]`
  - `cross_layer_search(self, query: str, layer_weights: Dict[str, float] = None) -> List[Dict[str, Any]]`
  - `_content_layer_search(self, layer, query: str) -> List[Tuple[str, float]]`
  - `_structure_layer_search(self, layer, query: str) -> List[Tuple[str, float]]`
  - `_reference_layer_search(self, layer, query: str) -> List[Tuple[str, float]]`
  - `_attribute_layer_search(self, layer, query: str) -> List[Tuple[str, float]]`
  - `path_based_search(self, start_node: str, end_node: str, ...)`
  - `connectivity_search(self, nodes: List[str], ...)`
  - `community_detection(self, algorithm: str = 'louvain', ...)`
  - `hypergraph_query(self, query_type: str, **kwargs) -> List[Dict[str, Any]]`
    - `traverse_diffusion(node, distance=0)` (nested function)
  - `multilayer_hypergraph_fusion(self, query: str, ...)`
- **Class: InfoRetriever**
  - `__init__(self, query_processor: QueryProcessor, rerank_processor: RerankProcessor, ...)`
  - `vector_retrieval(self, query: str, corpus: List[Dict[str, Any]], ...)`
  - `graph_retrieval(self, query: str, corpus: List[Dict[str, Any]], ...)`
  - `hybrid_retrieval(self, query: str, corpus: List[Dict[str, Any]], ...)`

### `topologist.py`
- **Class: PromptTopology**
  - `__init__(self, generator=None)`
  - `prompt_disambiguation(self, prompt: str, model_name: str, model_type: str, ...)`
  - `prompt_genetic_algorithm(self, initial_prompt: str, model_name: str, model_type: str, ...)`
  - `prompt_differential()`
  - `prompt_breeder()`
  - `prompt_phrase_evolution()`
  - `prompt_persona_search()`
  - `prompt_examplar()`
  - `prompt_reasoning(self, prompt: str, model_name: str, model_type: str, ...)`
- **Class: ScalingTopology**
  - `__init__(self, generator=None)`
  - `best_of_n_synthesis(self, prompt: str, model_name: str, model_type: str, ...)`
  - `best_of_n_selection(self, prompt: str, model_name: str, model_type: str, ...)`
  - `self_reflection(self, prompt: str, model_name: str, model_type: str, ...)`
  - `atom_of_thought(self, prompt: str, model_name: str, model_type: str, ...)`
  - `multimodel_debate_solo(self, prompt: str, model_configs: List[Dict[str, Any]], ...)`
  - `multimodel_debate_dual(self, prompt: str, model_configs: List[Dict[str, Any]], ...)`
  - `multipath_disambiguation_selection(self, prompt: str, model_name: str, model_type: str, ...)`
  - `socratic_dialogue(self, initial_prompt: str, model_name: str, model_type: str, ...)`
  - `hierarchical_decomposition(self, main_task_prompt: str, model_name: str, model_type: str, ...)`
    - `topological_sort(graph)` (nested function)
      - `visit(node_id)` (nested function)
  - `regenerative_majority_synthesis(self, prompt: str, model_name: str, model_type: str, ...)`
  - `adaptive_dag_reasoning(self, initial_prompt: str, model_name: str, model_type: str, ...)`
    - `process_hierarchical_node(node, parent_id=None, path="")` (nested function)
    - `topological_sort(dag)` (nested function)
    - `add_hierarchical_solutions(group_id, indent="")` (nested function)
  - `recursive_chain_of_thought(self, initial_prompt: str, model_name: str, model_type: str, ...)`
  - `ensemble_weighted_voting(self, prompt: str, model_configs: List[Dict[str, Any]], ...)`
  - `_calculate_consistency_score(self, responses: List[str]) -> float`
  - `_jaccard_similarity(self, str1: str, str2: str) -> float`
  - `_weighted_selection(self, weights: List[float]) -> int`

### `utility.py`
- **Class: DataUtility**
  - `ensure_directory(directory: Union[str, Path]) -> Path`
  - `text_operation(operation: str, file_path: Union[str, Path], data: Optional[Any] = None, ...)`
  - `csv_operation(operation: str, file_path: Union[str, Path], ...)`
  - `format_conversion(data: Any, output_format: str, **kwargs) -> Any`
  - `dataframe_operations(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame`
- **Class: StatisticsUtility**
  - `set_random_seed(size: int = 1, min_value: int = 0, max_value: int = 2**32-1) -> Union[int, List[int]]`
- **Class: AIUtility**
  - `_load_prompts(cls)`
  - `get_meta_prompt(cls, application: Optional[Literal["metaprompt","metaresponse", "metaworkflow"]] = "metaprompt", ...)`
  - `apply_meta_prompt(cls, application: Optional[Literal["metaprompt","metaresponse", "metaworkflow"]] = "metaprompt", ...)`
  - `get_meta_fix(cls, fix_template: str, fix_type: Optional[str] = None, component: Optional[str] = None) -> Union[Dict[str, str], Optional[str]]`
  - `get_task_prompt(cls, prompt_id: str) -> Optional[Dict]`
  - `list_prompt_categories(cls) -> List[str]`
    - `collect_categories(d: Dict, prefix: str = "")` (nested function)
  - `list_task_prompts(cls) -> List[Dict[str, Any]]`
  - `_get_encoder(cls, encoding_name: str = "cl100k_base")`
  - `process_tokens(cls, text: Union[str, List[str]], operation: str, encoding_name: str = "cl100k_base", **kwargs) -> Any`
  - `format_prompt_components(cls, prompt_id: str) -> Optional[str]`
  - `format_text_list(cls, texts: List[str], text_type: str = "prompt") -> str`
  - `format_json_response(cls, response: str) -> Dict[str, Any]`
  - `get_prompt_core_components(cls, prompt_json: Dict[str, Any]) -> Dict[str, Any]`
  - `merge_response_format(cls, prompt_json: Dict[str, Any], prompt_id: str) -> Dict[str, Any]`
