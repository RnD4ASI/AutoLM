# Auto GAI Coding Standards

## Overview

This document establishes coding standards and best practices for the Auto GAI project. Consistent adherence to these guidelines ensures code readability, maintainability, and quality across the codebase.

## Python Style Guidelines

### PEP 8 Conformance

All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards with the following specifics:

- **Line Length**: Maximum 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Whitespace**:
  - No trailing whitespace
  - Two blank lines before top-level class and function definitions
  - One blank line before method definitions within a class
  - Use blank lines sparingly within functions to indicate logical sections

Recommended tools:
- Use `flake8` or `pylint` for code style checking
- Configure your IDE to highlight PEP 8 violations

### Naming Conventions

- **Modules**: Short, lowercase names (e.g., `utility.py`, `generator.py`)
- **Packages**: Short, lowercase names, no underscores (e.g., `optimisation`)
- **Classes**: CapWords/CamelCase convention (e.g., `TextParser`, `VectorBuilder`)
- **Functions & Methods**: lowercase words with underscores (e.g., `get_completion`, `process_text`)
- **Variables**:
  - lowercase words with underscores (e.g., `embedding_model`, `max_tokens`)
  - Prefix private variables and methods with a single underscore (e.g., `_private_method`)
  - Prefix class-private members with double underscores (e.g., `__very_private`)
- **Constants**: ALL_CAPS_WITH_UNDERSCORES (e.g., `MAX_RECURSION_DEPTH`)

### Import Organization

Organize imports into groups separated by a blank line in this order:

1. Standard library imports
2. Related third-party imports (external packages)
3. Local application/library-specific imports

Example:
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party packages
import numpy as np
import pandas as pd
from loguru import logger
from openai import OpenAI

# Local modules
from src.utility import AIUtility, DataUtility
from src.logging import get_logger
```

Additional rules:
- Use absolute imports rather than relative imports
- Avoid wildcards (`from module import *`)
- Alphabetize imports within each group
- Prefer importing specific functions/classes when only a few are needed

### Type Annotation Usage

Type annotations are required for all public methods and functions. Internal methods should have annotations where they add clarity.

- Use Python's typing module for comprehensive type hints
- Include return type annotations, even for `None` returns
- For methods that may return different types, use Union types
- Annotate class variables in `__init__` methods

Example:
```python
def get_bleu(self, reference_text: str, generated_text: str, n: int = 4, 
             mode: str = 'calculation') -> Union[float, Dict[str, Any]]:
    """Calculate BLEU score between reference and generated text."""
    # function implementation
```

For complex return types, consider using:
- `TypedDict` for dictionary returns with specific keys
- `NamedTuple` or dataclasses for structured return values
- Proper generic annotations for collections

### Comment Style and Documentation

- Each module, class, method, and function must have a docstring
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include parameter and return type descriptions in docstrings
- Document exceptions raised and when they are raised
- Use inline comments sparingly and only to explain *why*, not *what*

Example docstring:
```python
def decompose_query(self, query: str, max_subqueries: int = 3) -> List[str]:
    """Breaks down a complex query into simpler subqueries.
    
    Args:
        query: The original query string to decompose
        max_subqueries: Maximum number of subqueries to generate
        
    Returns:
        A list of decomposed query strings
        
    Raises:
        ValueError: If query is empty or max_subqueries < 1
    """
```

## Test Requirements

### Test Coverage Expectations

- **Minimum coverage**: 80% overall code coverage
- **Core modules**: 90% or higher coverage for critical modules:
  - `generator.py`
  - `evaluator.py`
  - `retriever.py`
  - `topologist.py`
  - `dbbuilder.py`
- Unit tests for all public methods and functions
- Integration tests for module interactions
- Edge case testing, including error handling
- Manual verification required for UI/UX components

Run coverage reports regularly and include them in PRs.

### Test Structure and Naming Conventions

- Use `pytest` as the testing framework
- Organize tests to mirror the package structure
- Filename format: `test_<module_name>.py`
- Test class format: `Test<ClassUnderTest>`
- Test method format: `test_<method_name>_<scenario>`

Example:
```
/tests
  /unit
    /src
      test_generator.py  # Tests for src/generator.py
      test_evaluator.py  # Tests for src/evaluator.py
  /integration
    test_retrieval_pipeline.py  # Tests integration between components
  /performance
    test_large_vector_queries.py  # Performance-specific tests
```

Method naming examples:
```python
# Testing normal operation
def test_get_completion_with_valid_input()

# Testing edge cases
def test_get_completion_with_empty_prompt()
def test_get_completion_with_max_tokens_exceeded()

# Testing error conditions
def test_get_completion_raises_error_when_api_unavailable()
```

### Fixture Usage Guidelines

- Use fixtures for test setup and common operations
- Create fixtures at the appropriate scope (function, class, module, session)
- Use parameterized fixtures for testing multiple input scenarios
- Create fixtures for:
  - Database connections
  - API clients
  - Test data generation
  - Environment setup/teardown
  - Mock objects

Example fixture usage:
```python
@pytest.fixture(scope="module")
def vector_database():
    """Creates a test vector database with sample embeddings."""
    db = VectorDatabase("memory", dimension=384)
    # Add test vectors
    vectors = [np.random.rand(384) for _ in range(10)]
    texts = [f"Test document {i}" for i in range(10)]
    db.add_items(vectors, texts)
    yield db
    # Cleanup
    db.clear()

def test_semantic_search(vector_database):
    processor = VectorDBRetrievalProcessor(vector_database)
    results = processor.semantic_search("test query", top_k=3)
    assert len(results) == 3
```

### Mocking and Test Isolation

- Use `unittest.mock` or `pytest-mock` for external dependencies
- Mock expensive operations (API calls, database operations)
- Use dependency injection to facilitate mocking
- Create isolated tests that don't depend on external services

Example:
```python
def test_azure_completion(mocker):
    # Mock the OpenAI API client
    mock_completion = mocker.patch("src.generator.Generator._call_azure_openai")
    mock_completion.return_value = {
        "choices": [{"text": "Mocked response"}]
    }
    
    generator = Generator()
    result = generator.get_completion(
        prompt_id=1, 
        prompt="Test prompt", 
        model="gpt-4"
    )
    
    assert mock_completion.called
    assert "Mocked response" in result
```

## Continuous Integration

- All PRs must pass automated checks before merging:
  - Linting (flake8/pylint)
  - Type checking (mypy)
  - Unit tests pass
  - Coverage requirements met
- Test performance impact for changes to critical paths

---

These standards should be applied to all new code and gradually applied to existing code during refactoring. Deviations from these standards require justification and team approval.
