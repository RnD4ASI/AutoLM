from src.logging import get_logger
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, TypeVar
from pathlib import Path
import tiktoken
import random

logger = get_logger(__name__)

class DataUtility:
    """Static utility class for data operations."""

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """Create directory if it does not exist.
        
        Args:
            directory: Directory path to create
            
        Returns:
            Path object for created directory
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def text_operation(operation: str, file_path: Union[str, Path], data: Optional[Any] = None, 
                      file_type: str = 'text', **kwargs) -> Optional[Any]:
        """Perform text file operations (load/save/delete).
        
        Args:
            operation: One of 'load', 'save', 'delete'
            file_path: Path to the text file
            data: Data to save (required for 'save' operation)
            file_type: One of 'text' or 'json'
            **kwargs: Additional arguments for json operations (e.g. indent)
            
        Returns:
            Loaded data for 'load', True for successful 'delete', None for 'save'
            
        Raises:
            ValueError: If operation or file_type is invalid
            FileNotFoundError: If file does not exist for 'load'
        """
        path = Path(file_path)
        
        if operation == 'delete':
            if path.exists():
                path.unlink()
                return True
            return False
            
        elif operation == 'load':
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
                
            if file_type == 'text':
                return path.read_text()
            elif file_type == 'json':
                return json.loads(path.read_text())
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        elif operation == 'save':
            if data is None:
                raise ValueError("Data is required for save operation")
                
            path.parent.mkdir(parents=True, exist_ok=True)
            if file_type == 'text':
                path.write_text(str(data))
            elif file_type == 'json':
                path.write_text(json.dumps(data, **kwargs))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @staticmethod
    def csv_operation(operation: str, file_path: Union[str, Path], 
                     data: Optional[pd.DataFrame] = None, **kwargs) -> Optional[Union[pd.DataFrame, bool]]:
        """Perform CSV file operations (load/save/delete).
        
        Args:
            operation: One of 'load', 'save', 'delete'
            file_path: Path to the CSV file
            data: DataFrame to save (required for 'save' operation)
            **kwargs: Additional arguments for pandas read_csv/to_csv
            
        Returns:
            DataFrame for 'load', True for successful 'delete', None for 'save'
            
        Raises:
            ValueError: If operation is invalid
            FileNotFoundError: If file does not exist for 'load'
        """
        path = Path(file_path)
        
        if operation == 'delete':
            if path.exists():
                path.unlink()
                return True
            return False
            
        elif operation == 'load':
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return pd.read_csv(path, **kwargs)
            
        elif operation == 'save':
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame for save operation")
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, **kwargs)
            
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @staticmethod
    def format_conversion(data: Any, output_format: str, **kwargs) -> Any:
        """Convert data between different formats (dict/dataframe/string).
        
        Args:
            data: Data to convert (string in markdown format, DataFrame, or dict)
            output_format: One of 'dict', 'dataframe', 'string'
            **kwargs: Format-specific parameters:
                - dict_format: For dataframe to dict conversion (e.g. 'records', 'list')
                - headers: Optional list of column headers for markdown table
                - alignment: Optional list of column alignments ('left', 'center', 'right')
                
        Returns:
            Converted data in the specified output format
            
        Raises:
            ValueError: If format is invalid or markdown table is malformed
        """
        # Auto-detect input type
        input_format = None
        if isinstance(data, str):
            input_format = 'string'
        elif isinstance(data, pd.DataFrame):
            input_format = 'dataframe'
        elif isinstance(data, dict):
            input_format = 'dict'
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
            
        # Convert input to intermediate DataFrame format
        df = None
        if input_format == 'string':
            # Parse markdown table
            lines = [line.strip() for line in data.split('\n') if line.strip()]
            if not lines or '|' not in lines[0]:
                raise ValueError("Input string must be a markdown table with '|' separators")
            
            # Extract headers
            headers = [col.strip() for col in lines[0].strip('|').split('|')]
            headers = [h.strip() for h in headers if h.strip()]
            
            # Skip separator line if present (e.g., |:---:|:---:|)
            start_idx = 1
            if len(lines) > 1 and ':---:' in lines[1] or '---' in lines[1]:
                start_idx = 2
            
            # Parse data rows
            rows = []
            for line in lines[start_idx:]:
                if '|' not in line:
                    continue
                values = [val.strip() for val in line.strip('|').split('|')]
                values = [v.strip() for v in values if v.strip()]
                if len(values) == len(headers):
                    rows.append(values)
            
            df = pd.DataFrame(rows, columns=headers)
            
        elif input_format == 'dataframe':
            df = data
            
        elif input_format == 'dict':
            # Check if all values are scalar (not lists, arrays, or dicts)
            all_scalar = all(not isinstance(val, (list, dict, tuple, set, pd.Series, pd.DataFrame, pd.Index)) 
                            for val in data.values())
            
            if all_scalar and 'orient' not in kwargs:
                # Handle scalar values by default using 'index' orientation
                # This will create a DataFrame with the dict keys as the index
                # and a single column containing the values
                df = pd.DataFrame.from_dict(data, orient='index', columns=[0])
            else:
                # Use the provided kwargs or default handling for non-scalar values
                df = pd.DataFrame.from_dict(data, **kwargs)
            
        # Convert DataFrame to output format
        if output_format == 'dict':
            dict_format = kwargs.get('dict_format', 'records')
            return df.to_dict(dict_format)
            
        elif output_format == 'dataframe':
            return df
            
        elif output_format == 'string':
            # Convert DataFrame to markdown table
            headers = kwargs.get('headers', df.columns.tolist())
            alignments = kwargs.get('alignment', ['center'] * len(headers))
            
            # Validate alignments
            align_map = {
                'left': ':---',
                'center': ':---:',
                'right': '---:'
            }
            separators = [align_map.get(a.lower(), ':---:') for a in alignments]
            
            # Build markdown table
            lines = []
            
            # Add headers
            lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
            
            # Add separator line with alignments
            lines.append('| ' + ' | '.join(separators) + ' |')
            
            # Add data rows
            for _, row in df.iterrows():
                values = [str(row[col]) for col in df.columns]
                lines.append('| ' + ' | '.join(values) + ' |')
            
            return '\n'.join(lines)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    @staticmethod
    def dataframe_operations(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply a sequence of operations to a DataFrame.
        
        Args:
            df: Input DataFrame
            operations: List of operation dictionaries, each with:
                - 'type': Operation type
                - 'params': Operation parameters
                
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If operation type is unsupported
        """
        result = df.copy()
        
        for op in operations:
            op_type = op['type']
            params = op['params']
            
            if op_type == 'filter':
                result = result[result[params['column']].isin(params['values'])]
                
            elif op_type == 'sort':
                result = result.sort_values(**params)
                
            elif op_type == 'group':
                result = result.groupby(params['by']).agg(params['agg']).reset_index()
                
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
                
        return result

class StatisticsUtility:
    """Static utility class for statistical operations."""
    @staticmethod
    def set_random_seed(size: int = 1, min_value: int = 0, max_value: int = 2**32-1) -> Union[int, List[int]]:
        """Generate and set unique random seeds within a specified range.
        
        Args:
            size: Number of unique random seeds to generate. Default is 1.
            min_value: Minimum value for random seeds (inclusive). Default is 0.
            max_value: Maximum value for random seeds (inclusive). Default is 2**32-1.
            
        Returns:
            If size is 1, returns a single random seed as an integer.
            If size > 1, returns a list of unique random seeds.
            
        Raises:
            ValueError: If size is greater than the range of possible values,
                      or if min_value is greater than max_value.
        """
        if min_value > max_value:
            raise ValueError(f"min_value ({min_value}) must be less than or equal to max_value ({max_value})")
            
        possible_values = max_value - min_value + 1
        if size > possible_values:
            raise ValueError(
                f"Cannot generate {size} unique seeds in range [{min_value}, {max_value}] "
                f"(only {possible_values} possible values)"
            )
            
        if size == 1:
            # Generate a single seed
            seed = random.randrange(min_value, max_value + 1)
            return seed
        else:
            # Generate multiple unique seeds
            seeds = set()
            while len(seeds) < size:
                seeds.add(random.randrange(min_value, max_value + 1))
            return list(seeds)

class AIUtility:
    """Static utility class for AI-related operations."""
    
    _encoders = {}  # Cache for different encoders
    _meta_templates = None
    _task_chains = None
    
    @classmethod
    def _load_prompts(cls):
        """Load prompt templates and task chains if not already loaded."""
        if cls._meta_templates is None or cls._task_chains is None:
            try:
                config_dir = Path.cwd() / "config"
                cls._meta_templates = DataUtility.text_operation('load', config_dir / "meta_prompt_library.json", file_type='json')
                cls._task_chains = DataUtility.text_operation('load', config_dir / "task_prompt_library.json", file_type='json')
                logger.info("Prompt templates and task chains loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load prompt files: {e}")
                raise

    @classmethod
    def get_meta_prompt(cls, application: Optional[Literal["metaprompt","metaresponse"]] = "metaprompt", category: str = None, action: str = None) -> Optional[str]:
        """Get a meta prompt template by application, category and action.
        
        Args:
            application: Application area (e.g. 'metaprompt', 'metaresponse')
            category: Category of the template
            action: Specific action within the category
            
        Returns:
            Template string if found, None otherwise
        """
        cls._load_prompts()
        try:
            return cls._meta_templates[application][category][action]['template']
        except KeyError:
            logger.error(f"Meta template not found: application={application}, category={category}, action={action}")
            return None

    @classmethod
    def get_meta_fix(cls, fix_template: str, fix_type: Optional[str] = None, component: Optional[str] = None) -> Union[Dict[str, str], Optional[str]]:
        """Get the component modifications for a specific template and fix type.
        
        Args:
            fix_template: The reasoning template to use (e.g., "chain_of_thought")
            fix_type: The type of modification (prefix, postfix, replace)
            component: Optional specific component to retrieve (e.g., 'task', 'instruction', 'response_format')
            
        Returns:
            If component is specified, returns the modification string for that component.
            Otherwise, returns a dictionary mapping component names to their modifications.
            Returns None or empty dict if not found.
        """
        cls._load_prompts()
        try:
            # If a specific component is requested, return just that component's modification
            if component:
                return cls._meta_templates["ppfix"][fix_template][fix_type][component]
            
            elif fix_type:
                return cls._meta_templates["ppfix"][fix_template][fix_type]
            else:
                return cls._meta_templates["ppfix"][fix_template]
        
        except KeyError:
            if component:
                logger.warning(f"Meta fix not found: template={fix_template}, fix_type={fix_type}, component={component}")
                return None
            else:
                logger.warning(f"Meta fix not found: template={fix_template}, fix_type={fix_type}")
                return {}

    @classmethod
    def get_task_prompt(cls, prompt_id: str) -> Optional[Dict]:
        """Get a task prompt by its ID.
        
        Args:
            prompt_id: ID of the task prompt
            
        Returns:
            Task prompt dictionary if found, None otherwise
        """
        cls._load_prompts()
        try:
            return cls._task_chains['task_prompts'][prompt_id]
        except KeyError:
            logger.error(f"Task prompt not found: prompt_id={prompt_id}")
            return None

    @classmethod
    def list_prompt_categories(cls) -> List[str]:
        """List all available prompt categories with full paths.
        
        Returns:
            List of category paths (e.g. ['code/review', 'text/summarize'])
        """
        cls._load_prompts()
        categories = []
        
        def collect_categories(d: Dict, prefix: str = ""):
            for k, v in d.items():
                full_path = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    if 'template' in v:
                        categories.append(full_path)
                    else:
                        collect_categories(v, full_path)
        
        collect_categories(cls._meta_templates)
        return sorted(categories)

    @classmethod
    def list_task_prompts(cls) -> List[Dict[str, Any]]:
        """List all available task prompts with their basic info.
        
        Returns:
            List of dictionaries containing prompt ID and description
        """
        cls._load_prompts()
        return [
            {
                'id': prompt_id,
                'description': prompt['description']
            }
            for prompt_id, prompt in cls._task_chains['task_prompts'].items()
        ]

    @classmethod
    def _get_encoder(cls, encoding_name: str = "cl100k_base"):
        """Initialize or return cached tiktoken encoder.
        
        Args:
            encoding_name: Name of the encoding to use. Common options:
                - "o200k_base" (used by GPT-4o)
                - "cl100k_base" (default, used by GPT-4, GPT-3.5)
                - "p50k_base" (used by GPT-3)
        
        Returns:
            Tiktoken encoder instance
        """
        if encoding_name not in cls._encoders:
            cls._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        return cls._encoders[encoding_name]

    @classmethod
    def process_tokens(cls, text: Union[str, List[str]], operation: str, encoding_name: str = "cl100k_base", **kwargs) -> Any:
        """Process text with token operations.
        
        Args:
            text: Input text or texts
            operation: One of 'count', 'truncate', 'split'
            encoding_name: Name of the encoding to use (default: "cl100k_base")
            **kwargs: Operation-specific parameters
            
        Returns:
            Processed text or token counts
        """
        encoder = cls._get_encoder(encoding_name)
        
        if operation == 'count':
            if isinstance(text, str):
                return len(encoder.encode(text))
            return [len(encoder.encode(t)) for t in text]
            
        elif operation == 'truncate':
            max_tokens = kwargs.get('max_tokens', 100)
            if isinstance(text, str):
                tokens = encoder.encode(text)[:max_tokens]
                return encoder.decode(tokens)
            return [encoder.decode(encoder.encode(t)[:max_tokens]) for t in text]
            
        elif operation == 'split':
            chunk_size = kwargs.get('chunk_size', 100)
            overlap = kwargs.get('overlap', 0)
            
            if isinstance(text, str):
                tokens = encoder.encode(text)
                chunks = []
                for i in range(0, len(tokens), chunk_size - overlap):
                    chunk = tokens[i:i + chunk_size]
                    chunks.append(encoder.decode(chunk))
                return chunks
            return [cls.process_tokens(t, 'split', encoding_name=encoding_name, 
                                          chunk_size=chunk_size, overlap=overlap) for t in text]
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @classmethod
    def format_prompt_components(cls, prompt_id: str) -> Optional[str]:
        """Format prompt components into a structured string.
        
        Args:
            prompt_id: ID of the task prompt
            
        Returns:
            Formatted string with prompt components if found, None otherwise
            
        Example output:
            [Role]: Senior Software Engineer
            [Task]: Review the provided Python code for quality...
            [Purpose]: Ensure code quality, identify potential issues...
            ...
        """
        cls._load_prompts()
        
        if not cls._task_chains:
            logger.error("Task chains not loaded")
            return None
            
        # Get the prompt data
        prompt_data = cls._task_chains.get(prompt_id)
        if not prompt_data:
            logger.error(f"Prompt not found: {prompt_id}")
            return None
            
        # Extract components
        components = prompt_data.get('components', {})
        if isinstance(components, str):  # Handle string format (comma-separated)
            components = {
                'components': components
            }
            
        # Build formatted string
        formatted_parts = []
        
        # Add non-component fields first
        for field in ['promptid', 'name', 'description']:
            if field in prompt_data:
                formatted_parts.append(f"[{field.title()}]: {prompt_data[field]}")
        
        # Add components
        for key, value in components.items():
            if key != 'components':  # Skip the raw components string if present
                formatted_parts.append(f"[{key.replace('_', ' ').title()}]: {value}")
                
        return "\n".join(formatted_parts) if formatted_parts else None

    @classmethod
    def format_text_list(cls, texts: List[str], text_type: str = "prompt") -> str:
        """Format a list of prompts or responses into a single string.
        
        Args:
            texts: List of text strings (prompts or responses)
            text_type: Type of text being formatted ('prompt' or 'response')
            
        Returns:
            A formatted string with numbered entries
            
        Example outputs:
            Prompt 1: <prompt text>
            Prompt 2: <prompt text>
            ...
            
            Response 1: <response text>
            Response 2: <response text>
            ...
            
        Raises:
            ValueError: If text_type is not 'prompt' or 'response'
        """
        if text_type.lower() not in ["prompt", "response"]:
            raise ValueError("text_type must be either 'prompt' or 'response'")
            
        # Capitalize first letter for display
        display_type = text_type.capitalize()
        
        # Format each text with its number
        formatted_texts = [
            f"{display_type} {i+1}: {text.strip()}"
            for i, text in enumerate(texts)
        ]
        
        # Join with newlines
        return "\n\n".join(formatted_texts)

    @classmethod
    def format_json_response(cls, response: str) -> Dict[str, Any]:
        """Format a JSON response string into a dictionary.
        
        Args:
            response: JSON response string
            
        Returns:
            Dictionary representation of the JSON string
            
        Raises:
            JSONDecodeError: If the input string is not valid JSON
        """
        try:
            tmp = json.loads(response.strip('```json').strip('```'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
        return tmp

    @classmethod
    def get_prompt_core_components(cls, prompt_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all components from a prompt JSON except response_format and iteration.
        
        Args:
            prompt_json: Prompt in JSON format following task_prompt_library.json structure
            
        Returns:
            Dictionary containing all components except response_format and iteration
            
        Example input:
            {
                "role": "Senior Engineer",
                "task": "Review code",
                "response_format": "List format",
                "iteration": "yes"
            }
            
        Example output:
            {
                "role": "Senior Engineer",
                "task": "Review code"
            }
        """
        try:
            # Create a deep copy to avoid modifying original
            result = json.loads(json.dumps(prompt_json))
            
            # Remove response_format and iteration if they exist
            result.pop("response_format", None)
            result.pop("iteration", None)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting prompt components: {e}")
            raise

    @classmethod
    def merge_response_format(cls, prompt_json: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
        """Merge a prompt JSON with response_format from task_prompt_library.json.
        
        Args:
            prompt_json: Prompt in JSON format without response_format
            prompt_id: ID of the prompt in task_prompt_library.json to get response_format from
            
        Returns:
            Merged prompt JSON with response_format added from reference prompt
            
        Example:
            Input prompt_json:
                {
                    "role": "Engineer",
                    "task": "Review code"
                }
            
            Reference prompt (prompt_id="prompt_001"):
                {
                    "promptid": 1,
                    "components": {
                        "response_format": "List format with severity"
                    }
                }
                
            Output:
                {
                    "role": "Engineer",
                    "task": "Review code",
                    "response_format": "List format with severity"
                }
        """
        try:
            # Load reference prompt
            cls._load_prompts()
            if not cls._task_chains or prompt_id not in cls._task_chains:
                raise ValueError(f"Prompt ID {prompt_id} not found in task chains")
            
            reference_prompt = cls._task_chains[prompt_id]
            
            # Get response_format from reference prompt
            if ("components" in reference_prompt and 
                "response_format" in reference_prompt["components"]):
                response_format = reference_prompt["components"]["response_format"]
            else:
                raise ValueError(f"No response_format found for prompt ID {prompt_id}")
            
            # Create deep copy of input prompt
            task_prompt = json.loads(json.dumps(prompt_json))
            
            # Add response_format directly to the dictionary
            task_prompt["response_format"] = response_format
            
            return task_prompt
            
        except Exception as e:
            logger.error(f"Error merging prompt response format: {e}")
            raise