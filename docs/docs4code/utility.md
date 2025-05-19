# Utility Module Documentation

The utility module provides essential helper functions and utilities for data manipulation, statistical operations, and AI-related tasks.

## Table of Contents
- [Data Utility](#data-utility)
- [Statistics Utility](#statistics-utility)
- [AI Utility](#ai-utility)

## Data Utility

The `DataUtility` class provides static methods for file and data operations.

### Methods

#### `ensure_directory(directory: Union[str, Path]) -> Path`
Creates a directory if it doesn't exist.
```python
path = DataUtility.ensure_directory("/path/to/directory")
```

#### `load_file(file_path: Union[str, Path], file_type: str = 'text', **kwargs) -> Any`
Loads file content based on type.
- Supported types: 'text', 'json', 'table'
```python
# Load JSON file
data = DataUtility.load_file("config.json", file_type="json")

# Load CSV with pandas options
df = DataUtility.load_file("data.csv", file_type="table", index_col=0)
```

#### `save_file(file_path: Union[str, Path], data: Any, file_type: str = 'text', **kwargs) -> None`
Saves data to file based on type.
```python
# Save dictionary as JSON
DataUtility.save_file("config.json", config_dict, file_type="json", indent=4)

# Save DataFrame as CSV
DataUtility.save_file("data.csv", df, file_type="table", index=False)
```

#### `delete_file(file_path: Union[str, Path]) -> bool`
Deletes a file if it exists.
```python
success = DataUtility.delete_file("old_file.txt")
```

#### `dataframe_operations(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame`
Applies multiple operations to a DataFrame.
```python
operations = [
    {
        'type': 'filter',
        'params': {'column': 'category', 'values': ['A', 'B']}
    },
    {
        'type': 'sort',
        'params': {'by': 'value', 'ascending': False}
    }
]
result_df = DataUtility.dataframe_operations(df, operations)
```

## Statistics Utility

The `StatisticsUtility` class provides methods for statistical operations and random sampling.

### Methods

#### `set_random_seed(seed: Optional[int] = None) -> int`
Sets random seed for reproducibility.
```python
seed = StatisticsUtility.set_random_seed(42)
```

#### `sample_data(items: List[Any], params: Dict[str, Any]) -> Union[List[Any], np.ndarray]`
Samples data using various methods.
```python
# Random sampling
samples = StatisticsUtility.sample_data(
    items=[1, 2, 3, 4, 5],
    params={'method': 'random', 'size': 3}
)

# Weighted sampling
weighted_samples = StatisticsUtility.sample_data(
    items=['A', 'B', 'C'],
    params={
        'method': 'weighted',
        'size': 2,
        'weights': [0.5, 0.3, 0.2]
    }
)
```

#### `generate_sequence(size: int, params: Dict[str, Any]) -> np.ndarray`
Generates number sequences from various distributions.
```python
# Normal distribution
normal_seq = StatisticsUtility.generate_sequence(
    size=100,
    params={
        'distribution': 'normal',
        'mean': 0,
        'std': 1
    }
)

# Uniform distribution
uniform_seq = StatisticsUtility.generate_sequence(
    size=50,
    params={
        'distribution': 'uniform',
        'low': 0,
        'high': 10
    }
)
```

## AI Utility

The `AIUtility` class provides methods for token-based text processing using the tiktoken library.

### Methods

#### `process_tokens(text: Union[str, List[str]], operation: str, **kwargs) -> Union[int, List[int], str, List[str]]`
Processes text with token operations.
```python
# Count tokens
token_count = AIUtility.process_tokens(
    "Sample text",
    operation="count"
)

# Truncate text
truncated = AIUtility.process_tokens(
    "Long text...",
    operation="truncate",
    max_tokens=100
)

# Split into chunks
chunks = AIUtility.process_tokens(
    "Long document...",
    operation="split",
    chunk_size=500,
    overlap=50
)
```

#### `batch_process_tokens(texts: List[str], operation: str, batch_size: int = 32, **kwargs) -> List[Any]`
Processes multiple texts in batches for better performance.
```python
results = AIUtility.batch_process_tokens(
    texts=["text1", "text2", "text3"],
    operation="count",
    batch_size=2
)
```

## Integration Example

Here's an example showing how these utilities work together:

```python
# 1. Set up directories and random seed
data_dir = DataUtility.ensure_directory("data/")
StatisticsUtility.set_random_seed(42)

# 2. Generate and save sample data
sequence = StatisticsUtility.generate_sequence(
    size=1000,
    params={'distribution': 'normal', 'mean': 0, 'std': 1}
)
df = pd.DataFrame({'values': sequence})
DataUtility.save_file("data/samples.csv", df, file_type="table")

# 3. Load and process data
df = DataUtility.load_file("data/samples.csv", file_type="table")
operations = [
    {'type': 'filter', 'params': {'column': 'values', 'values': [-1, 1]}},
    {'type': 'sort', 'params': {'by': 'values'}}
]
processed_df = DataUtility.dataframe_operations(df, operations)

# 4. Process text with tokens
documents = ["doc1", "doc2", "doc3"]
token_counts = AIUtility.batch_process_tokens(
    texts=documents,
    operation="count"
)
```

## Error Handling

All utility classes implement robust error handling:

1. **File Operations**:
   - FileNotFoundError for missing files
   - OSError for file operation failures
   - ValueError for unsupported file types

2. **Statistical Operations**:
   - ValueError for invalid distribution types
   - ValueError for unsupported sampling methods

3. **AI Operations**:
   - ImportError for missing tiktoken package
   - ValueError for unsupported token operations

## Best Practices

1. **File Handling**:
   - Always use `ensure_directory` before saving files
   - Use appropriate file types for data format
   - Handle potential file operation errors

2. **Statistical Operations**:
   - Set random seed for reproducibility
   - Choose appropriate sampling methods
   - Validate distribution parameters

3. **Token Processing**:
   - Use batch processing for multiple texts
   - Consider token limits when truncating
   - Handle potential tokenization errors
