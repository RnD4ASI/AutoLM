This document breaks down the sequence of interactions in the `evaluator.py` module.

```mermaid
sequenceDiagram
    participant Client
    participant Evaluator
    participant RetrievalEvaluator
    participant Generator
    participant NLP_Models
    
    %% Initialization
    Client->>Evaluator: __init__(generator)
    Evaluator->>spacy: load("en_core_web_sm")
    Evaluator->>Rouge: Initialize()
    
    %% Text Evaluation Flow
    Client->>Evaluator: evaluate_text(generated_text, reference_text, metrics=["bleu", "rouge", "bertscore"])
    
    %% BLEU Score Calculation
    Evaluator->>Evaluator: get_bleu(reference_text,generated_text)
    Evaluator->>Evaluator: _generate_ngrams(tokens,n)
    Evaluator-->>Evaluator: bleu_score
    
    %% ROUGE Score Calculation
    Evaluator->>Evaluator: get_rouge(reference_text, generated_text)
    Evaluator->>Rouge: get_scores()
    Rouge-->>Evaluator: rouge_scores
    
    %% BERTScore Calculation
    Evaluator->>Evaluator: get_bertscore(reference_text, generated_text)
    Evaluator->>Generator: get_embeddings(text, model)
    Generator-->>Evaluator: embeddings
    Evaluator->>Evaluator: cosine_similarity(embeddings)
    
    %% Return Combined Results
    Evaluator-->>Client: evaluation_metrics
    
    %% Retrieval Evaluation Flow
    Client->>RetrievalEvaluator: evaluate_retrieval_quality(query, retrieved_docs, relevant_docs)
    
    %% Calculate Metrics
    RetrievalEvaluator->>RetrievalEvaluator: _calculate_rank_metrics()
    RetrievalEvaluator->>RetrievalEvaluator: _calculate_ndcg()
    RetrievalEvaluator->>RetrievalEvaluator: _calculate_map()
    RetrievalEvaluator-->>Client: retrieval_metrics
    
    %% Analyze Term Overlap
    RetrievalEvaluator->>RetrievalEvaluator: _analyze_term_overlap(query,docs)
    RetrievalEvaluator->>RetrievalEvaluator: _estimate_query_effectiveness()
    
    %% Return Retrieval Evaluation
    RetrievalEvaluator-->>Client: retrieval_metrics
    
    %% Error Handling
    alt Error in any operation
        Evaluator/RetrievalEvaluator->>Evaluator/RetrievalEvaluator: Log error
        Evaluator/RetrievalEvaluator-->>Client: Raise appropriate exception
    end
```

## Key Components and Interactions

1. **Evaluator**: Main class for text quality evaluation
   - Implements BLEU, ROUGE, and BERTScore calculations
   - Handles text tokenization and preprocessing
   - Manages model initialization and text processing

2. **RetrievalEvaluator**: Specialized evaluator for retrieval systems
   - Implements ranking metrics (NDCG, MAP)
   - Analyzes term overlap and query effectiveness
   - Evaluates retrieval performance

3. **Generator**: Provides text embeddings for semantic evaluation
   - Used by Evaluator for BERTScore calculations
   - Handles model inference for embeddings

## Main Workflows

### Text Quality Evaluation
1. Tokenizes input texts
2. Calculates BLEU scores for n-gram precision
3. Computes ROUGE scores for recall-oriented metrics
4. Generates BERT embeddings for semantic similarity
5. Returns comprehensive metrics dictionary

### Retrieval System Evaluation
1. Processes query and document sets
2. Calculates precision, recall, and F1 scores
3. Computes ranking metrics (NDCG, MAP)
4. Analyzes term overlap between queries and documents
5. Estimates query effectiveness

## Error Handling
- Validates input formats and parameters
- Handles model loading errors gracefully
- Provides detailed error messages
- Implements fallback mechanisms for missing metrics
