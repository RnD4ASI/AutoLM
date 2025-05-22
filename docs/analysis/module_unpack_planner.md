This document breaks down the sequence of interactions in the `planner.py` module, focusing on both external interactions and internal function calls within the TaskPlanner class.

```mermaid
sequenceDiagram
    participant Client
    participant TaskPlanner
    participant Shortlist
    participant ExtractData
    participant Sequence
    participant Normalize
    participant Optimize
    participant Prepare
    participant MetaGenerator
    participant DataUtility

    %% Main flow
    Client->>+TaskPlanner: plan_task_sequence()
    
    %% Step 1: Load task prompt library
    TaskPlanner->>+DataUtility: text_operation('load', library_path)
    DataUtility-->>-TaskPlanner: task_library
    
    %% Internal TaskPlanner operations
    rect rgba(200, 200, 200, 0.1)
        Note over TaskPlanner: Internal Operations
        
        TaskPlanner->>+Shortlist: _shortlist_task_prompt()
        Shortlist->>MetaGenerator: get_meta_generation(select_task_prompts)
        MetaGenerator-->>Shortlist: selection_response
        Shortlist-->>-TaskPlanner: selected_prompt_ids
        
        TaskPlanner->>+ExtractData: _extract_selected_prompts_data()
        ExtractData-->>-TaskPlanner: selected_prompts_data
        
        TaskPlanner->>+Sequence: _sequence_task_prompt()
        Sequence->>MetaGenerator: get_meta_generation(sequence_task_prompts)
        MetaGenerator-->>Sequence: sequence_response
        
        Sequence->>+Normalize: _normalise_prompt_flow_config()
        Normalize-->>-Sequence: normalized_config
        
        Sequence->>+Optimize: _optimize_edges()
        Optimize-->>-Sequence: optimized_edges
        
        Sequence-->>-TaskPlanner: prompt_flow_config
        
        TaskPlanner->>+Prepare: _prepare_shortlist_task_prompt()
        Prepare-->>-TaskPlanner: shortlisted_prompts
    end
    
    %% Return results
    TaskPlanner-->>-Client: (prompt_flow_config, shortlisted_prompts)
    
    %% Optional: Save results
    alt Save Results
        Client->>+TaskPlanner: save_planning_results()
        TaskPlanner->>+DataUtility: text_operation('save', flow_path)
        DataUtility-->>-TaskPlanner: success
        TaskPlanner->>+DataUtility: text_operation('save', prompts_path)
        DataUtility-->>-TaskPlanner: success
        TaskPlanner-->>-Client: success
    end
    
    %% Optional: User feedback
    alt User Feedback
        Client->>+TaskPlanner: incorporate_user_feedback()
        TaskPlanner->>TaskPlanner: Apply feedback modifications
        TaskPlanner-->>-Client: (updated_flow_config, updated_prompts)
    end
```

## Key Components and Interactions

1. **TaskPlanner**: Main class that orchestrates the planning process
   - `_shortlist_task_prompt()`: Selects relevant prompts based on the goal
   - `_sequence_task_prompt()`: Sequences the selected prompts into an execution flow
   - `_normalise_prompt_flow_config()`: Validates and normalizes the flow configuration
   - `_prepare_shortlist_task_prompt()`: Prepares the final list of shortlisted prompts
   - `_extract_selected_prompts_data()`: Extracts full details of selected prompts
   - `_optimize_edges()`: Optimizes the flow graph for better parallelization
   - `_infer_task_type()`: Determines the type of task for a given prompt
   - `_select_best_selection_candidate()`: Chooses the best prompt selection from candidates
   - `_select_best_sequencing_candidate()`: Chooses the best sequence from candidates
   - `_generate_prompt_flow_config()`: Creates the prompt flow configuration
   - `_create_shortlisted_prompts()`: Creates the shortlisted prompts dictionary
   - `_calculate_complexity_score()`: Computes a complexity score for goals
   - `_basic_complexity_analysis()`: Provides fallback complexity analysis

2. **MetaGenerator**: Handles reasoning-based prompt selection and sequencing
3. **DataUtility**: Manages file I/O operations for loading/saving configurations
4. **AIUtility & Evaluator**: Support components for AI-related operations and evaluation

## Main Workflow

1. **Initialization**: The TaskPlanner is initialized with configuration and required utilities
2. **Prompt Shortlisting**: 
   - Loads task prompt library
   - Uses reasoning model to select relevant prompts based on the goal
3. **Prompt Sequencing**:
   - Sequences the shortlisted prompts into an optimal execution flow
   - Normalizes and optimizes the flow configuration
4. **Result Handling**:
   - Prepares the final shortlisted prompts dictionary
   - Optionally saves results to files
   - Handles user feedback for refinement

## Error Handling
- Falls back to basic selection/sequencing if reasoning model fails
- Includes validation steps to ensure configuration integrity
- Handles missing or invalid prompt IDs gracefully
- Provides fallback mechanisms for complexity analysis and task type inference
