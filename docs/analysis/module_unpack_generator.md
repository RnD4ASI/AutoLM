This document breaks down the sequence of interactions in the `generator.py` module.

```mermaid
sequenceDiagram
    participant Client
    participant Generator
    participant MetaGenerator
    participant Encoder
    participant AzureOpenAI
    participant HuggingFace
    
    %% Main Generation Flow
    Client->>Generator: get_completion(prompt_id, prompt, model=model_name, temperature=0.2, max_tokens=4000, ...)
    
    %% Model Selection and Validation
    Generator->>Generator: _validate_model(model_name, "completion")
    
    alt Azure OpenAI Model
        Generator->>Generator: refresh_token()
        Generator->>AzureOpenAI: POST /completions
        AzureOpenAI-->>Generator: completion_response
    else HuggingFace Model
        Generator->>HuggingFace: AutoModelForCausalLM.generate()
        HuggingFace-->>Generator: generated_text
    end
    
    %% Post-processing
    Generator->>Generator: _post_process_response()
    Generator-->>Client: completion_result
    
    %% Meta Generation Flow
    Client->>MetaGenerator: get_meta_generation(application="metaprompt", category="reasoning", action="select_task_prompts", prompt_id=1001, ...)
    
    MetaGenerator->>AIUtility: get_meta_prompt_template()
    AIUtility-->>MetaGenerator: template
    
    MetaGenerator->>Generator: get_completion(prompt=formatted_template, model=meta_model)
    Generator-->>MetaGenerator: meta_response
    MetaGenerator-->>Client: meta_generation_result
    
    %% Encoding Flow
    Client->>Encoder: encode(texts)
    
    alt Bi-Encoder
        Encoder->>SentenceTransformer: encode()
        SentenceTransformer-->>Encoder: embeddings
    else Cross-Encoder/Reranker
        Encoder->>CrossEncoder: predict()
        CrossEncoder-->>Encoder: similarity_scores
    end
    
    Encoder-->>Client: encoded_output
    
    %% Error Handling
    alt Error in any operation
        Generator/MetaGenerator/Encoder->>Generator: Log error
        Generator/MetaGenerator/Encoder-->>Client: Raise appropriate exception
    end
```

## Key Components and Interactions

1. **Generator**: Main class for text generation
   - Handles model selection and validation
   - Manages API calls to different providers
   - Implements retry logic and error handling

2. **MetaGenerator**: Specialized generator for meta-prompting
   - Retrieves and formats meta-prompt templates
   - Handles structured responses using JSON schemas
   - Integrates with the base Generator for completions

3. **Encoder**: Handles text encoding and similarity
   - Supports bi-encoders for dense embeddings
   - Implements cross-encoders for document ranking
   - Manages model loading and inference

## Main Workflows

### Text Generation
1. Validates model and parameters
2. Routes to appropriate provider (Azure OpenAI or HuggingFace)
3. Processes and returns the generated text

### Meta-Prompting
1. Retrieves and formats meta-prompt template
2. Fills in template placeholders
3. Generates structured response
4. Validates and parses the output

### Text Encoding
1. Loads appropriate encoder model
2. Processes input text
3. Returns embeddings or similarity scores

## Error Handling
- Validates model names and parameters
- Implements retry logic for API calls
- Provides detailed error logging
- Falls back to alternative models when possible
