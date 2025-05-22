This document breaks down the sequence of interactions in the `topologist.py` module.

```mermaid
sequenceDiagram
    participant Client
    participant PromptTopology
    participant MetaGenerator
    participant Generator
    participant Evaluator
    
    %% Initialization
    Client->>PromptTopology: __init__(generator)
    PromptTopology->>MetaGenerator: Initialize
    PromptTopology->>Evaluator: Initialize
    
    %% Prompt Disambiguation Flow
    Client->>PromptTopology: prompt_disambiguation(task_prompt, prompt_id)
    
    %% Step 1: Generate disambiguated prompt
    PromptTopology->>MetaGenerator: get_meta_generation(application="metaprompt", category="evaluation", action="disambiguate", task_prompt=task_prompt)
    MetaGenerator-->>PromptTopology: disambiguation
    
    %% Step 2: Rewrite prompt based on disambiguation
    PromptTopology->>MetaGenerator: get_meta_generation(application="metaprompt", category="manipulation", action="rewrite", task_prompt=task_prompt, feedback=disambiguation)
    MetaGenerator-->>PromptTopology: rewritten_prompt
    
    %% Step 3: Execute both original and rewritten prompts
    par Original Prompt
        PromptTopology->>Generator: get_completion(task_prompt)
        Generator-->>PromptTopology: response_pre
    and Rewritten Prompt
        PromptTopology->>Generator: get_completion(rewritten_prompt)
        Generator-->>PromptTopology: response_post
    end
    
    %% Return results
    PromptTopology-->>Client: (rewritten_prompt, response_post) or full history
    
    %% Genetic Algorithm Flow
    Client->>PromptTopology: prompt_genetic_algorithm(base_prompt, population_size=10, generations=5)
    
    loop For each generation
        PromptTopology->>PromptTopology: _create_population()
        PromptTopology->>Generator: get_completion(mutated_prompts)
        Generator-->>PromptTopology: responses
        PromptTopology->>Evaluator: evaluate_responses()
        Evaluator-->>PromptTopology: fitness_scores
        PromptTopology->>PromptTopology: _select_and_crossover()
        PromptTopology->>PromptTopology: _mutate_population()
        
        loop num_variations - 1
            PromptTopology->>MetaGenerator: get_meta_generation(action="rephrase", task_prompt=task_prompt)
            MetaGenerator-->>PromptTopology: variation
        end
    end
    
    %% Step 2: Evaluate variations
    loop For each variation
        PromptTopology->>Generator: get_completion(variation)
        Generator-->>PromptTopology: response
        PromptTopology->>Evaluator: evaluate_response(response)
        Evaluator-->>PromptTopology: score
    end
    
    %% Step 3: Evolve through iterations
    loop num_evolution times
        %% Crossover and mutation
        PromptTopology->>PromptTopology: _crossover_and_mutate(top_variations)
        
        %% Evaluate new generation
        loop For each new variant
            PromptTopology->>Generator: get_completion(new_variant)
            Generator-->>PromptTopology: response
            PromptTopology->>Evaluator: evaluate_response(response)
            Evaluator-->>PromptTopology: score
        end
    end
    
    %% Return best performing prompt and response
    PromptTopology-->>Client: (best_prompt, best_response) or full history
    
    %% Error Handling
    alt Error in any operation
        Component->>Component: Log error
        Component-->>Client: Return error or fallback
    end
```

## Key Components and Interactions

1. **PromptTopology**: Main class implementing various prompt engineering strategies
   - Manages the execution flow of different topologies
   - Coordinates between generators, evaluators, and other components
   - Handles error cases and fallback mechanisms

2. **MetaGenerator**: Handles meta-level prompt operations
   - Generates variations of prompts
   - Performs disambiguation and rewriting
   - Manages prompt templates and patterns

3. **Generator**: Core text generation component
   - Executes prompts using various models
   - Handles model-specific parameters and configurations
   - Manages response generation and formatting

4. **Evaluator**: Assesses prompt and response quality
   - Provides feedback on response quality
   - Scores variations for genetic algorithm
   - Supports multiple evaluation metrics

## Main Workflows

### Prompt Disambiguation
1. Analyze input prompt for ambiguity
2. Generate disambiguation suggestions
3. Rewrite prompt based on analysis
4. Execute both original and rewritten prompts
5. Return improved results

### Genetic Algorithm
1. Generate initial population of prompt variations
2. Evaluate each variation's performance
3. Select top performers for reproduction
4. Apply crossover and mutation
5. Repeat for specified number of generations
6. Return best performing prompt

### Regenerative Majority Synthesis (from Memory)
1. Generate multiple initial responses
2. Truncate and regenerate responses
3. Apply majority vote or synthesis
4. Use parameters: num_initial_responses, num_regen_responses, cut_off_fraction, synthesis_method

## Error Handling
- Validates input prompts and parameters
- Handles API failures and timeouts
- Provides fallback mechanisms for failed generations
- Logs detailed error information for debugging
