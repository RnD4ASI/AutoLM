# APRA Analysis System Module Documentation

This document provides a comprehensive overview of the core modules in the APRA Analysis System.

## Table of Contents
- [Database Builder](#database-builder)
- [Evaluator](#evaluator)
- [Generator](#generator)
- [Prompt Operations](#prompt-operations)
- [Response Operations](#response-operations)
- [Retriever](#retriever)

## Database Builder
The `dbbuilder.py` module handles the creation and management of knowledge bases.

### DatabaseBuilder
Main class for building and managing knowledge bases.

#### Methods
- `build_vector_db(documents: List[str]) -> Dict[str, Any]`
  - Creates a vector database from input documents
  - Generates embeddings and stores document metadata
  ```python
  db_builder = DatabaseBuilder()
  documents = ["doc1.txt", "doc2.txt"]
  vector_db = db_builder.build_vector_db(documents)
  ```

- `build_graph_db(documents: List[str]) -> Dict[str, Any]`
  - Creates a graph database with entity relationships
  - Extracts entities and builds connections
  ```python
  graph_db = db_builder.build_graph_db(documents)
  ```

## Evaluator
The `evaluator.py` module provides response evaluation capabilities.

### ResponseEvaluator
Evaluates the quality and effectiveness of responses.

#### Methods
- `evaluate_response(response: str, context: Dict[str, Any]) -> Dict[str, Any]`
  - Evaluates response quality using multiple criteria
  - Returns detailed evaluation metrics
  ```python
  evaluator = ResponseEvaluator()
  result = evaluator.evaluate_response(
      response="The solution is...",
      context={"query": "How to..."}
  )
  ```

- `generate_criteria(task: str) -> Dict[str, Any]`
  - Generates custom evaluation criteria for specific tasks
  ```python
  criteria = evaluator.generate_criteria("Explain quantum computing")
  ```

## Generator
The `generator.py` module handles text generation and embeddings.

### Generator
Manages interactions with language models for text generation and embeddings.

#### Methods
- `get_completion(prompt: str, **kwargs) -> Dict[str, Any]`
  - Generates text completions using language models
  ```python
  generator = Generator()
  response = generator.get_completion(
      prompt="Explain how...",
      temperature=0.7
  )
  ```

- `get_embedding(text: str) -> np.ndarray`
  - Generates embeddings for input text
  ```python
  embedding = generator.get_embedding("Sample text")
  ```

## Prompt Operations
The `promptops.py` module handles prompt manipulation and optimization.

### PromptOperator
Manages prompt transformation and optimization.

#### Methods
- `transform_to_program(task: str, additional_context: Optional[Dict[str, Any]]) -> Dict[str, Any]`
  - Converts tasks into program specifications
  ```python
  prompt_op = PromptOperator(generator, prompt_library)
  program_spec = prompt_op.transform_to_program(
      task="Create a sorting function",
      additional_context={"language": "python"}
  )
  ```

- `optimize_prompt(prompt: str, optimization_type: str) -> Dict[str, Any]`
  - Optimizes prompts for better results
  ```python
  optimized = prompt_op.optimize_prompt(
      prompt="How to...",
      optimization_type="clarity"
  )
  ```

## Response Operations
The `responseops.py` module handles response generation and refinement.

### ResponseOperator
Manages response generation and improvement.

#### Methods
- `generate_response(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]`
  - Generates responses using context and prompt
  ```python
  response_op = ResponseOperator(generator)
  response = response_op.generate_response(
      prompt="Explain...",
      context={"background": "..."}
  )
  ```

- `refine_response(response: str, feedback: Dict[str, Any]) -> Dict[str, Any]`
  - Refines responses based on feedback
  ```python
  refined = response_op.refine_response(
      response="Initial answer...",
      feedback={"clarity": "needs improvement"}
  )
  ```

## Retriever
The `retriever.py` module handles document retrieval and knowledge base querying.

### Retriever
Manages document retrieval using multiple methods.

#### Methods
- `retrieve(query: str, context: Optional[Dict[str, Any]], retrieval_type: str = "hybrid", scoring_method: str = "semantic") -> List[Dict[str, Any]]`
  - Retrieves relevant documents using various methods
  ```python
  retriever = Retriever(generator, knowledge_base_path="kb/")
  results = retriever.retrieve(
      query="What is...",
      retrieval_type="hybrid",
      scoring_method="hyde"
  )
  ```

- `_vector_retrieve(query: str) -> List[Dict[str, Any]]`
  - Performs vector-based retrieval
- `_symbolic_retrieve(query: str, method: str) -> List[Dict[str, Any]]`
  - Performs symbolic retrieval (TF-IDF, BM25)
- `_hyde_retrieve(query: str) -> List[Dict[str, Any]]`
  - Performs Hypothetical Document Embeddings retrieval

### Features
- Multiple retrieval methods:
  - Semantic (vector-based)
  - Symbolic (TF-IDF, BM25)
  - HyDE (Hypothetical Document Embeddings)
- Query processing:
  - Query rewriting
  - Query decomposition
- Reranking methods:
  - Reciprocal Rank Fusion
  - Cross-encoder reranking

## Integration Example
Here's an example showing how these components work together:

```python
# Initialize components
generator = Generator()
db_builder = DatabaseBuilder()
retriever = Retriever(generator, "knowledge_base/")
evaluator = ResponseEvaluator(generator)
prompt_op = PromptOperator(generator, prompt_library)
response_op = ResponseOperator(generator)

# Process a query
query = "Explain the impact of climate change on marine ecosystems"

# 1. Retrieve relevant documents
results = retriever.retrieve(
    query=query,
    retrieval_type="hybrid",
    scoring_method="hyde"
)

# 2. Generate response
response = response_op.generate_response(
    prompt=query,
    context={"documents": results}
)

# 3. Evaluate response
evaluation = evaluator.evaluate_response(
    response=response["content"],
    context={"query": query}
)

# 4. Refine if needed
if evaluation["overall_score"] < 0.8:
    refined = response_op.refine_response(
        response=response["content"],
        feedback=evaluation
    )
```

This documentation provides an overview of the main components and their interactions in the APRA Analysis System. Each module is designed to work independently while also supporting seamless integration with other components.
