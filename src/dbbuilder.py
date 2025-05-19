from src.logging import get_logger
from typing import List, Any, Dict, Optional, Union, Tuple
import os
import subprocess
import spacy
import pandas as pd
import networkx as nx
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import networkx as nx
import glob
from io import BytesIO
import time
import traceback
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from src.utility import AIUtility
from src.generator import Generator, MetaGenerator
import json

logger = get_logger(__name__)


class TextParser:
    """
    Provides methods for cleansing data including PDF to Markdown conversion and text chunking.
    """
    def __init__(self) -> None:
        """Initialize a TextParser instance.
        
        No parameters required.
        
        Returns:
            None
        """
        # Initialize any resources, e.g., spaCy model if needed
        logger.debug("TextParser initialization started")
        try:
            start_time = time.time()
            self.nlp = spacy.load("en_core_web_sm")
            logger.debug(f"Loaded spaCy model in {time.time() - start_time:.2f} seconds")
            self.aiutility = AIUtility()
            self.generator = Generator()
            logger.debug("TextParser initialized successfully")
        except Exception as e:
            logger.error(f"TextParser initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise

    def pdf2md_markitdown(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using the MarkItDown package.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If MarkItDown package is not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_markitdown (MarkItDown) for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        try:
            from markitdown import MarkItDown
        except ImportError:
            logger.error("MarkItDown package not installed. Please install using: pip install markitdown")
            raise ImportError("MarkItDown package not installed")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize converter and convert PDF to Markdown
            start_time = time.time()
            logger.debug(f"Initializing MarkItDown converter")
            converter = MarkItDown()
            logger.debug(f"Starting PDF conversion: {pdf_path} -> {md_path}")
            converter.convert_file(pdf_path, md_path)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")

    def pdf2md_openleaf(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using the openleaf-markdown-pdf shell command.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            RuntimeError: If the openleaf-markdown-pdf command fails or isn't installed.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_openleaf (OpenLeaf) for {pdf_path}")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'

        try:
            # Run the openleaf-markdown-pdf command
            cmd = ['openleaf-markdown-pdf', '--input', pdf_path, '--output', md_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully converted PDF to Markdown: {md_path}")
                return None
            else:
                error_msg = result.stderr or "Unknown error occurred"
                logger.error(f"Command failed: {error_msg}")
                raise RuntimeError(f"PDF conversion failed: {error_msg}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute openleaf-markdown-pdf: {str(e)}")
            raise RuntimeError(f"openleaf-markdown-pdf command failed: {str(e)}")
        except FileNotFoundError:
            logger.error("openleaf-markdown-pdf command not found. Please install it first.")
            raise RuntimeError("openleaf-markdown-pdf is not installed")

    def pdf2md_ocr(self, pdf_path: str, md_path: str, model: str = "GOT-OCR2") -> None:
        """Convert a PDF to Markdown using an open sourced model from HuggingFace.
        tmp/ocr folder is used to store the temporary images.

        Parameters:
            pdf_path (str): The file path to the PDF.
            md_path (str): The file path to save the generated Markdown.
            model (str): The model to use for conversion (default is "GOT-OCR2").

        Returns:
            None

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If required packages are not installed.
            RuntimeError: If conversion fails.
        """
        # Convert PDF pages to images
        logger.info(f"Converting PDF to Markdown using pdf2md_ocr (OCR Model) for {pdf_path}")
        if pdf_path is None or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        pages = convert_from_path(pdf_path)
        if not pages:
            raise ValueError("No pages found in the PDF.")
        
        tmp_dir = Path.cwd() / "tmp/ocr"
        tmp_dir.mkdir(exist_ok=True)
        
        # Save each image temporarily and collect their file paths
        try:
            image_paths = []
            for idx, page in enumerate(pages):
                image_path = tmp_dir / f"temp_page_{idx}.jpg"
                page.save(image_path, "JPEG")
                image_paths.append(image_path)
            
            # Execute OCR on all temporary image files
            ocr_text = self.generator.get_hf_ocr(image_paths=image_paths, model=model)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            
            # Clean up temporary files
            for image_path in image_paths:
                os.remove(image_path)
            logger.info(f"PDF to Markdown conversion completed.")

        except Exception as e:
            logger.error(f"PDF to Markdown conversion failed: {str(e)}")
            raise e

    def pdf2md_llamaindex(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using LlamaIndex and PyMuPDF.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If required packages are not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_llamaindex for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        try:
            # Import required packages
            import pymupdf4llm
            from llama_index.core import Document
        except ImportError:
            logger.error("Required packages not installed. Please install using: pip install pymupdf4llm llama-index")
            raise ImportError("Required packages not installed: pymupdf4llm, llama-index")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize LlamaIndex processor options
            llamaindex_options = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'use_chunks': False
            }
            
            # Extract text using LlamaIndex and PyMuPDF
            start_time = time.time()
            logger.debug(f"Initializing LlamaIndexPDFProcessor with options: {llamaindex_options}")
            
            # Create a LlamaMarkdownReader with appropriate options
            import inspect
            reader_params = inspect.signature(pymupdf4llm.LlamaMarkdownReader.__init__).parameters
            
            # Prepare kwargs based on available parameters
            kwargs = {}
            
            # Add options if they are supported by the current version
            if 'chunk_size' in reader_params:
                kwargs['chunk_size'] = llamaindex_options['chunk_size']
                
            if 'chunk_overlap' in reader_params:
                kwargs['chunk_overlap'] = llamaindex_options['chunk_overlap']
            
            # Create reader with configured options
            logger.debug(f"Creating LlamaMarkdownReader with parameters: {kwargs}")
            llama_reader = pymupdf4llm.LlamaMarkdownReader(**kwargs)
            
            # Load and convert the PDF to LlamaIndex documents
            load_data_params = inspect.signature(llama_reader.load_data).parameters
            load_kwargs = {}
            
            # Add any additional load_data parameters if supported
            if 'use_chunks' in load_data_params:
                load_kwargs['use_chunks'] = llamaindex_options['use_chunks']
                
            logger.debug(f"Loading PDF with parameters: {load_kwargs}")
            documents = llama_reader.load_data(str(pdf_path), **load_kwargs)
            
            # Combine all documents into a single markdown text
            if llamaindex_options['use_chunks']:
                # Return documents as they are (already chunked by LlamaIndex)
                markdown_text = "\n\n---\n\n".join([doc.text for doc in documents])
            else:
                # Combine all text into a single document
                markdown_text = "\n\n".join([doc.text for doc in documents])
            
            # Write the markdown output to file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown using LlamaIndex: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")
    
    def pdf2md_pymupdf(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using PyMuPDF directly.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If pymupdf4llm package is not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_pymupdf for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        try:
            # Import required packages
            import pymupdf4llm
            import inspect
        except ImportError:
            logger.error("pymupdf4llm package not installed. Please install using: pip install pymupdf4llm")
            raise ImportError("pymupdf4llm package not installed")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize PyMuPDF processor options
            pymupdf_options = {
                'preserve_images': False,
                'preserve_tables': True
            }
            
            start_time = time.time()
            logger.debug(f"Using PyMuPDF with options: {pymupdf_options}")
            
            # Use pymupdf4llm to convert directly to markdown
            # Check pymupdf4llm version to see if it supports the options
            to_markdown_params = inspect.signature(pymupdf4llm.to_markdown).parameters
            
            # Prepare kwargs based on available parameters
            kwargs = {}
            
            # Add options if they are supported by the current version
            if 'preserve_images' in to_markdown_params:
                kwargs['preserve_images'] = pymupdf_options['preserve_images']
                
            if 'preserve_tables' in to_markdown_params:
                kwargs['preserve_tables'] = pymupdf_options['preserve_tables']
                
            # Call to_markdown with appropriate options
            logger.debug(f"Converting PDF to Markdown with parameters: {kwargs}")
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path), **kwargs)
            
            # Write the markdown output to file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown using PyMuPDF: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")
    
    def pdf2md_textract(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using the textract library.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If textract package is not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_textract for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        try:
            import textract
        except ImportError:
            logger.error("textract package not installed. Please install using: pip install textract")
            raise ImportError("textract package not installed")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize textract options
            textract_options = {
                'method': 'pdftotext',
                'encoding': 'utf-8',
                'layout': True
            }
            
            start_time = time.time()
            logger.debug(f"Using textract with options: {textract_options}")
            
            # Build the extraction options
            extract_kwargs = {
                'method': textract_options['method'],
                'encoding': textract_options['encoding'],
                'layout': textract_options['layout']
            }
            
            # Remove None values
            extract_kwargs = {k: v for k, v in extract_kwargs.items() if v is not None}
            
            # Extract text from PDF
            logger.debug(f"Extracting text with parameters: {extract_kwargs}")
            text = textract.process(str(pdf_path), **extract_kwargs).decode(textract_options['encoding'])
            
            # Convert plain text to basic markdown
            # This is a simple conversion since textract doesn't preserve formatting well
            lines = text.split('\n')
            markdown_lines = []
            in_paragraph = False
            
            for line in lines:
                line = line.strip()
                if not line:  # Empty line
                    if in_paragraph:
                        markdown_lines.append('')  # End paragraph
                        in_paragraph = False
                else:
                    # Very basic heuristic for headings: all caps, not too long
                    if line.isupper() and len(line) < 100:
                        markdown_lines.append(f"## {line}")
                        in_paragraph = False
                    else:
                        if not in_paragraph:
                            in_paragraph = True
                        markdown_lines.append(line)
            
            markdown_text = '\n'.join(markdown_lines)
            
            # Write the markdown output to file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown using textract: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")


class TextChunker:
    """Handles text chunking and markdown processing operations.
    
    This class provides methods for splitting markdown documents into chunks using different strategies:
    1. Length-based chunking: Divides text into fixed-size chunks with configurable overlap
    2. Hierarchy-based chunking: Divides text based on heading structure and document hierarchy
    
    The chunking parameters (chunk size and overlap) can be configured through the main_config.json file
    or passed directly to the chunking methods. Default values are used as fallbacks.
    
    Attributes:
        nlp: spaCy language model for text processing
        default_chunk_size: Default size of chunks in characters (from config or 1000)
        default_chunk_overlap: Default overlap between chunks in characters (from config or 100)
    """
    
    def __init__(self, config_file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize TextChunker with default configuration from main_config.json.
        
        Parameters:
            config_file_path (Optional[Union[str, Path]]): Path to the configuration file.
                If None, defaults to Path.cwd() / "config" / "main_config.json".
                
        Returns:
            None
        """
        logger.debug("TextChunker initialization started")
        try:
            start_time = time.time()
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load defaults from config file
            chunker_config_path = Path(config_file_path) if config_file_path else Path.cwd() / "config" / "main_config.json"
            try:
                if os.path.exists(chunker_config_path):
                    with open(chunker_config_path, 'r') as f:
                        config = json.load(f)
                    # Get chunking configuration from main_config.json
                    chunker_config = config.get('knowledge_base', {}).get('chunking', {}).get('other_config', {})
                    self.default_chunk_size = chunker_config.get('chunk_size', 1000)
                    self.default_chunk_overlap = chunker_config.get('chunk_overlap', 100)
                    logger.debug(f"Loaded chunking configuration from {chunker_config_path}: chunk_size={self.default_chunk_size}, chunk_overlap={self.default_chunk_overlap}")
                else:
                    logger.warning(f"Config file not found at {chunker_config_path}, using default values")
                    self.default_chunk_size = 1000
                    self.default_chunk_overlap = 100
                    
            except Exception as config_error:
                logger.error(f"Error loading config: {str(config_error)}, using default values")
                self.default_chunk_size = 1000
                self.default_chunk_overlap = 100
                
            build_time = time.time() - start_time
            logger.debug(f"Loaded spaCy model in {build_time:.2f} seconds")
            logger.debug("TextChunker initialized successfully")
        except Exception as e:
            logger.error(f"TextChunker initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def length_based_chunking(self, markdown_file: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> pd.DataFrame:
        """Chunks text from a markdown file into overlapping segments and returns as DataFrame.
        This is a pure length-based chunking method that doesn't consider headings.
        
        Parameters:
            markdown_file (str): Path to the markdown file.
            chunk_size (int, optional): Size of each chunk in characters. If None, uses the default_chunk_size.
            overlap (int, optional): Number of characters to overlap between chunks. If None, uses the default_chunk_overlap.
        
        Returns:
            pd.DataFrame: DataFrame containing:
                - reference: Unique identifier for each chunk
                - hierarchy: Document name (no heading hierarchy)
                - corpus: The chunk text content
        
        Raises:
            FileNotFoundError: If markdown file doesn't exist
            ValueError: If chunk_size <= overlap
        """
        # Use default values if not provided
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        if overlap is None:
            overlap = self.default_chunk_overlap
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
            
        # Get filename for reference generation
        filename = Path(markdown_file).stem.upper()
        logger.debug(f"Processing {filename} with chunk_size={chunk_size}, overlap={overlap}")
            
        # Read markdown file
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            logger.error(f"Markdown file not found: {markdown_file}")
            raise
            
        # Initialize chunks list
        chunks = []
        start = 0
        chunk_id = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate chunk boundaries
            end = min(start + chunk_size, text_length)
            
            # Get chunk text
            chunk_text = text[start:end].strip()
            
            # Only add chunk if it contains content
            if chunk_text:
                # Create chunk entry with appropriate field names for the graph builder
                chunk = {
                    'reference': f"{filename} Para {chunk_id + 1}.",
                    'hierarchy': filename,  # Just use filename, no heading hierarchy
                    'corpus': chunk_text
                }
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next chunk
            start = end - overlap
            
        # Convert to DataFrame
        df = pd.DataFrame(chunks)
        logger.info(f"Created {len(chunks)} chunks from {markdown_file}")
        return df

    def hierarchy_based_chunking(self, markdown_file: str, df_headings: pd.DataFrame) -> pd.DataFrame:
        """Extract hierarchical content chunks from a markdown file based on headings.
        
        Args:
            markdown_file: Path to the markdown file to process
            df_headings: DataFrame containing heading metadata with columns:
                - Level: Heading hierarchy level (e.g. 1, 2, 3)
                - Heading: Heading text
                - Page: Page number
                - File: File identifier (e.g. APS113)
                - Index: Index number
        
        Returns:
            DataFrame containing:
                - hierarchy: Full heading path (e.g. "APS 113 > Main Body > Application")
                - heading: Current heading text
                - reference: Document reference
                - corpus: Text content under the heading
                - content_type: Type of content ('paragraph', 'table', etc.)
        """
        try:
            # Extract filename and read content
            filename = Path(markdown_file).stem.upper()
            logger.debug(f"Processing {filename}")
            
            # Clean heading metadata
            df_clean = df_headings.copy()
            df_clean['Level'] = pd.to_numeric(df_clean['Level'], errors='coerce', downcast='integer')
            df_clean['Heading'] = df_clean['Heading'].str.strip()
            df_clean = df_clean.dropna(subset=['Level', 'Heading'])
            
            # Filter headings for current document
            doc_headings = df_clean[df_clean['File'].str.contains(filename, case=False, na=False)]
            doc_headings = doc_headings.sort_values(by='Index')
            if doc_headings.empty:
                logger.error(f"No matching headings found for {filename}")
                raise

            # Load markdown file
            with open(markdown_file, 'r', encoding='utf-8') as f:
                md_content = f.read().strip()
            
            # Extract content after the first heading
            first_heading = "## " + doc_headings.iloc[0]['Heading']
            first_heading_index = md_content.find(first_heading)
            if first_heading_index != -1:
                # Extract date from content before first heading if exists
                header_content = md_content[:first_heading_index]
                # Look for a specific date format for the commencement date
                # date_match = re.search(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', header_content)  # DD Month YYYY
                # if date_match:
                #     document_date = date_match.group()
                #     logger.debug(f"Found document date: {document_date}")
                # Extract the main content from the markdown file
                md_content = md_content[first_heading_index:]
                logger.debug(f"First heading {first_heading} found at index {first_heading_index}")
            else:
                md_content = ""
                logger.error(f"First heading {first_heading} not found in {filename}")
            
            # Split content into lines
            lines = md_content.split('\n')

            # Initialize variables
            extracted_sections = []
            current_heading = None
            current_content = []
            current_level = 0
            hierarchy = [] 
            
            # Process content line by line
            for i, line in enumerate(lines):                
                # Check if line is a heading
                if line.startswith('## '):
                    if current_heading:
                        self._append_section(filename = filename,
                                        hierarchy = hierarchy,
                                        current_heading = current_heading,
                                        current_content = current_content,
                                        extracted_sections = extracted_sections)
                    
                    current_heading = re.sub(r'\$.*\$', '', line[3:]).strip()
                    current_level = doc_headings[doc_headings['Heading'] == current_heading]['Level'].iloc[0]
                    doc_headings = doc_headings.drop(doc_headings.index[0])
                    
                    if current_level == 1:
                        if "Attachment" not in current_heading:
                            current_heading = filename + " - " + current_heading
                        hierarchy = [current_heading]
                    elif current_level >= 1:
                        if len(hierarchy) >= current_level:
                            hierarchy = hierarchy[:current_level-1]
                            hierarchy.append(current_heading)
                        else:
                            hierarchy.append(current_heading)  
                    current_content = []
                else:
                    if line.strip():
                        current_content.append(line.strip())
            
            # Append last section
            if current_heading:
                self._append_section(filename = filename,
                                hierarchy = hierarchy,
                                current_heading = current_heading,
                                current_content = current_content,
                                extracted_sections = extracted_sections)
            
            # Convert to DataFrame with proper column names
            if extracted_sections:
                df_sections = pd.DataFrame(extracted_sections, columns=['hierarchy', 'reference', 'heading', 'corpus'])
                # Add content_type column (default to 'paragraph')
                df_sections['content_type'] = 'paragraph'
                logger.info(f"Extracted {len(df_sections)} sections from {markdown_file}")
                return df_sections
            else:
                # Return empty DataFrame with required columns
                logger.warning(f"No sections extracted from {markdown_file}")
                return pd.DataFrame(columns=['hierarchy', 'heading', 'reference', 'corpus', 'content_type'])
            
        except FileNotFoundError:
            logger.error(f"Markdown file not found: {markdown_file}")
            raise
        except Exception as e:
            logger.error(f"Error processing markdown file {markdown_file}: {str(e)}")
            raise

    def _append_section(
        self, 
        filename: str, 
        hierarchy: List[str], 
        current_heading: str, 
        current_content: List[str],
        extracted_sections: List[Dict[str, str]]):
        """ Append a section to the extracted sections.
        
        Args:
            filename: Document identifier
            hierarchy: Current heading hierarchy
            current_heading: Heading level of the current section
            current_content: Regular paragraph content
            extracted_sections: List to append the new section to
            aiutility: Utility object for AI-based content extraction
        """
        if len(hierarchy) == 1:
            hierarchy_heading = hierarchy[0]
        else:
            hierarchy_heading = ' > '.join(hierarchy)
        
        if len(current_content) == 1:
            text_contents = current_content[0]
        else:
            text_contents = '\n'.join(current_content)
        
        attachment_prefix = re.search(r'Attachment ([A-Z])', hierarchy_heading) or \
                            re.search(r'Chapter \d+', hierarchy_heading) or \
                            re.search(r'CHAPTER \d+', hierarchy_heading)
        prefix_para = ", ".join([filename, attachment_prefix.group(0)]) if attachment_prefix else filename

        # Specific to Basel Framework
        if "CRE" in filename and text_contents is not None:
            # Split by numbering system N.N
            paragraphs = re.split(r'\n(?=\d+\.\d+)', text_contents)
            for paragraph in paragraphs:
                match = re.search(r'(\d+\.\d+)', paragraph)
                if match:
                    para_num = match.group(2)
                    ref_num = ", CRE" + match.group(0)
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para + " Para " + para_num + ref_num, 
                                                current_heading,
                                                paragraph.strip()))
                else:
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para+ " Orphan", 
                                                current_heading,
                                                paragraph.strip()))
                logger.debug(f"Paragraph found: {filename} - {current_heading.strip()}")

        # Specific to APS, APG and Risk Opinions
        else:
            # APS/APG - Split by numbering system N. or Table M
            if re.search(r'^(\d+\.)|(Table \d+)', text_contents):
                paragraphs = re.split(r'\n(?=\d+\.|Table \d+)', text_contents)
            # Split for long paragraphs (based on character count instead of token count)
            elif len(text_contents) > 8000:
                paragraphs = re.split(r'\n', text_contents)
            else:
                paragraphs = [text_contents]

            for paragraph in paragraphs:
                match1 = re.match(r'^(\d+\.)', paragraph)
                match2 = re.match(r'^Table (\d+)', paragraph)
                if match1:
                    logger.debug(f"Paragraph found: {filename} - {current_heading.strip()}")
                    para_num = match1.group(1)
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para + " Para " + para_num, 
                                                current_heading,
                                                paragraph.strip()))
                elif match2:
                    logger.debug(f"Table found: {filename} - {current_heading.strip()}")
                    table_num = match2.group(1)
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para + " Table " + table_num, 
                                                current_heading,
                                                paragraph.strip()))
                else:
                    logger.debug(f"Orphan paragraph: {filename} - {current_heading.strip()}")
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para+ " Orphan", 
                                                current_heading,
                                                paragraph.strip()))


class VectorBuilder:
    """
    Creates and manages a vector database for knowledge representation.
    Depends on TextParser and TextChunker for text parsing and chunking.
    
    Methods:
        create_vectordb: Creates a vector database from markdown files
        load_vectordb: Loads a vector database from a file
        merge_vectordbs: Merges multiple vector databases into one
    """
    def __init__(self, parser: TextParser, chunker: TextChunker, generator=None, config_file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize VectorBuilder with a DataCleanser instance.
        
        Parameters:
            parser (TextParser): Instance of TextParser.
            chunker (TextChunker): Instance of TextChunker.
            generator (Generator, optional): Instance of Generator. If None, a new one will be created.
            config_file_path (Optional[Union[str, Path]]): Path to the configuration file.
                If None, defaults to Path.cwd() / "config" / "main_config.json".
        
        Returns:
            None
        """
        # Store parser and chunker instances
        self.parser = parser
        self.chunker = chunker
        # Use provided generator or create a new one
        self.generator = generator if generator is not None else Generator()
        self.metagenerator = MetaGenerator(generator=self.generator)
        self.default_buffer_ratio = 0.9
        # Load defaults from config file
        chunker_config_path = Path(config_file_path) if config_file_path else Path.cwd() / "config" / "main_config.json"
        try:
            if os.path.exists(chunker_config_path):
                with open(chunker_config_path, 'r') as f:
                    config = json.load(f)
                # Get chunking configuration from main_config.json
                chunking_config = config.get('knowledge_base', {}).get('chunking', {}).get('other_config', {})
                self.default_chunk_size = chunking_config.get('chunk_size', 1000)
                self.default_chunk_overlap = chunking_config.get('chunk_overlap', 100)
                self.default_buffer_ratio = chunking_config.get('buffer_ratio', 0.9)
                logger.debug(f"VectorBuilder loaded chunking configuration from {actual_config_path}: chunk_size={self.default_chunk_size}, chunk_overlap={self.default_chunk_overlap}")
            else:
                logger.warning(f"Config file not found at {actual_config_path}, using default values")
                self.default_chunk_size = 1000
                self.default_chunk_overlap = 100
        except Exception as config_error:
            logger.error(f"Error loading config: {str(config_error)}, using default values")
            self.default_chunk_size = 1000
            self.default_chunk_overlap = 100
        
        logger.debug("VectorBuilder initialized successfully")

    def create_vectordb(self, 
                            markdown_file: str,
                            df_headings: pd.DataFrame,
                            chunking_method: str = 'hierarchy',
                            model: Optional[str] = None,
                            **kwargs) -> str:
        """Create a vector database from markdown files.
        
        Args:
            markdown_file: Path to the markdown file to process
            df_headings: DataFrame containing heading metadata
            chunking_method: Method to use for chunking ('hierarchy' or 'length')
            model: Model to use for embedding generation

            **kwargs: Additional arguments based on chunking method:

                For 'length' method:
                    - chunk_size: Size of chunks (default: 1000)
                    - chunk_overlap: Overlap between chunks (default: 100)
                For 'hierarchy' method:
                    - None
                    
        Returns:
            str: Path to the saved vector database file
        """
        try:
            # Validate chunking method
            if chunking_method not in ['hierarchy', 'length']:
                raise ValueError(f"Invalid chunking method: {chunking_method}")
            
            # Set defaults and validate parameters based on chunking method
            if chunking_method == 'length':
                chunk_size = kwargs.get('chunk_size', self.default_chunk_size)
                chunk_overlap = kwargs.get('chunk_overlap', self.default_chunk_overlap)
            else:  # hierarchy based chunking
                if any(k in kwargs for k in ['chunk_size', 'chunk_overlap']):
                    logger.warning("chunk_size and chunk_overlap are ignored for hierarchy-based chunking")
                
            
            # Get list of markdown files and hierarchy data
            if not markdown_file:
                raise ValueError(f"Markdown file not provided for vector database building")
            if not df_headings:
                raise ValueError(f"Hierarchy data not provided for vector database building")
            
            # Process each markdown file                
            # Choose chunking method
            if chunking_method == 'hierarchy':
                chunks_df = self.chunker.hierarchy_based_chunking(
                    markdown_file,
                    df_headings)
            else:
                chunks_df = self.chunker.length_based_chunking(
                    markdown_file,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )

            # Generate embeddings for corpus and hierarchy
            logger.info("Generating embeddings for corpus texts")
            buffer_ratio = kwargs.get('buffer_ratio', self.default_buffer_ratio)
            
            corpus_vectors = self.generator.get_embeddings(
                text=chunks_df['corpus'].tolist(),
                model=model,
                buffer_ratio=buffer_ratio
            )
            
            logger.info("Generating embeddings for hierarchy texts")
            hierarchy_vectors = self.generator.get_embeddings(
                text=chunks_df['hierarchy'].tolist(),
                model=model,
                buffer_ratio=buffer_ratio
            )
            
            # Add vector columns
            chunks_df['corpus_vector'] = corpus_vectors
            chunks_df['hierarchy_vector'] = hierarchy_vectors
            
            # Save to parquet file
            markdown_name_no_ext = os.path.splitext(os.path.basename(markdown_file))[0].lower()
            
            # Ensure using proper db directory path
            db_dir = Path.cwd() / 'db'
            db_dir.mkdir(exist_ok=True, parents=True)
            output_file = db_dir / f'vector_{markdown_name_no_ext}.parquet'
            chunks_df.to_parquet(output_file)
            
            logger.info(f"Vector database created with {len(chunks_df)} chunks and saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error creating vector database: {str(e)}")
            logger.debug(f"Vector database creation error details: {traceback.format_exc()}")
            raise

    def load_vectordb(self, parquet_file: str = None) -> pd.DataFrame:
        """Load the vector database from disk.
        
        Args:
            parquet_file: Path to the parquet file containing the vector database.
                          If None, will look for a file in the db directory.
            
        Returns:
            DataFrame containing the knowledge base with vector columns
            
        Raises:
            FileNotFoundError: If the vector database file doesn't exist
            ValueError: If the file format is not supported or the file is corrupted
        """
        logger.debug(f"Starting to load vector database from {parquet_file}")
        start_time = time.time()
        
        try:
            # If no filepath is provided, try to find a vector database in the db directory
            if parquet_file is None:
                db_dir = Path.cwd() / 'db'
                vector_files = list(db_dir.glob('vector_*.parquet'))
                if not vector_files:
                    raise FileNotFoundError("No vector database files found in the db directory")
                # Use the most recently modified file
                parquet_file = str(sorted(vector_files, key=lambda f: f.stat().st_mtime, reverse=True)[0])
                logger.debug(f"Using most recent vector database file: {parquet_file}")
            
            # Check if file exists
            if not os.path.exists(parquet_file):
                raise FileNotFoundError(f"Vector database not found: {parquet_file}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_file)
            
            # Validate required columns
            required_columns = ['reference', 'hierarchy', 'corpus']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in vector database")
            
            # Check for vector columns
            vector_columns = [col for col in df.columns if 'vector' in col or 'embedding' in col]
            if not vector_columns:
                logger.warning(f"No vector columns found in {parquet_file}")
            
            # Convert vector columns from string to numpy arrays if needed
            for col in vector_columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].apply(np.array)
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numpy arrays: {e}")
            
            logger.info(f"Loaded vector database from {parquet_file} with {len(df)} chunks in {time.time() - start_time:.2f} seconds")
            return df
            
        except Exception as e:
            logger.error(f"Error loading vector database: {str(e)}")
            logger.debug(f"Load error details: {traceback.format_exc()}")
            raise
    
    def merge_vectordbs(self, parquet_files: List[str], output_name: str = None) -> pd.DataFrame:
        """Merge multiple vector databases into one.
        
        Args:
            parquet_files: List of paths to vector database files to merge
            output_name: Name for the merged vector database file (without extension)
            
        Returns:
            pd.DataFrame: The merged vector database
            
        Raises:
            ValueError: If no parquet_files are provided or they are incompatible
            FileNotFoundError: If any of the specified files don't exist
        """
        logger.debug(f"Starting to merge {len(parquet_files)} vector databases")
        start_time = time.time()
        
        if not parquet_files:
            raise ValueError("No parquet files provided for merging")
            
        try:
            # Load and validate each dataframe
            dataframes = []
            for i, filepath in enumerate(parquet_files):
                logger.debug(f"Loading vector database {i+1}/{len(parquet_files)}: {filepath}")
                
                # Check if file exists
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Vector database file not found: {filepath}")
                
                # Load dataframe
                df = pd.read_parquet(filepath)
                
                # Validate required columns
                required_columns = ['reference', 'hierarchy', 'corpus']
                for col in required_columns:
                    if col not in df.columns:
                        logger.warning(f"Required column '{col}' not found in {filepath}, skipping")
                        continue
                
                # Check for vector columns
                vector_columns = [col for col in df.columns if 'vector' in col or 'embedding' in col]
                if not vector_columns:
                    logger.warning(f"No vector columns found in {filepath}, skipping")
                    continue
                
                # Add prefix to reference IDs to avoid conflicts
                prefix = f"db{i}_"
                df['reference'] = df['reference'].apply(lambda x: f"{prefix}{x}")
                
                # Add source file information
                df['source_file'] = os.path.basename(filepath)
                
                dataframes.append(df)
                logger.debug(f"Added {len(df)} chunks from {filepath}")
            
            if not dataframes:
                raise ValueError("No valid vector databases found to merge")
                
            # Merge dataframes
            merged_df = pd.concat(dataframes, ignore_index=True)
            
            # Generate output name if not provided
            if not output_name:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"merged_vectordb_{timestamp}"
                
            # Remove .parquet extension if included
            if output_name.endswith('.parquet'):
                output_name = output_name[:-8]
                
            # Save the merged vector database
            db_dir = Path.cwd() / 'db'
            db_dir.mkdir(exist_ok=True, parents=True)
            output_path = db_dir / f"vector_{output_name}.parquet"
            merged_df.to_parquet(output_path)
                
            logger.info(f"Merged {len(dataframes)} vector databases into {output_path} with {len(merged_df)} total chunks in {time.time() - start_time:.2f} seconds")
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to merge vector databases: {str(e)}")
            logger.debug(f"Merge error details: {traceback.format_exc()}")
            raise

    def _extract_references_with_lm(self, text: str, generator: Generator) -> List[str]:
        """Extract references from text using a language model.
        
        Args:
            text (str): Text containing potential references
            generator (Generator): Generator instance for language model inference
            
        Returns:
            List[str]: List of formatted references
        """
        try:
            # Define the application, category, and action for the prompt template
            # If this doesn't exist in prompt_templates.json yet, it will need to be added
            try:
                # Try to use a template from prompt_templates.json if it exists
                response = self.metagenerator.get_meta_generation(
                    application="extraction",
                    category="document",
                    action="references",
                    model="Qwen2.5-1.5B",
                    text=text,
                    temperature=0.1,  # Low temperature for consistent formatting
                    max_tokens=200
                )
            except ValueError:
                logger.warning("Reference extraction template not found in prompt_templates.json, using fallback prompt")
            
            # Validation of output - tbc
            return response
            
        except Exception as e:
            logger.warning(f"Language model reference extraction failed: {e}")
            return []

    def extract_reference(self, vectordb_file: str) -> pd.DataFrame:
        """Extract cross-references between corpus entries and add reference_additional column.
        
        This method identifies when one corpus entry references another and creates a
        reference_additional column that follows the same format as the reference column.
        This allows for linking corpus entries through a simple merge operation.
        
        Parameters:
            knowledge_base_path (str): Path to the knowledge base parquet file
        
        Returns:
            pd.DataFrame: Enhanced knowledge base with reference_additional column
        """
        try:
            # Load the knowledge base
            # find whether the vectordb_file exist in db folder using Path () and db folder path
            db_dir = Path.cwd() / 'db'
            vectordb_path = db_dir / vectordb_file
            if not vectordb_path.exists():
                raise FileNotFoundError(f"Knowledge base not found: {vectordb_path}")
            
            df = pd.read_parquet(vectordb_path)
            logger.info(f"Loaded knowledge base with {len(df)} entries")
            
            # Create a copy to avoid modifying the original
            enhanced_df = df.copy()
            
            # Initialize the reference_additional column as empty lists
            enhanced_df['reference_additional'] = [[] for _ in range(len(enhanced_df))]
            
            # Create a reference lookup dictionary for faster matching
            reference_lookup = {}
            for idx, row in enhanced_df.iterrows():
                # Extract filename from hierarchy (first part before the >)
                hierarchy_parts = row['hierarchy'].split(' > ')
                if hierarchy_parts:
                    filename = hierarchy_parts[0].split(' - ')[0].strip()
                    # Store the reference with its index
                    reference_lookup[row['reference']] = idx
            
            # Initialize Generator for language model
            try:
                use_lm = True
                logger.info("Initialized language model for reference extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize language model: {e}")
                use_lm = False
            
            # Common reference patterns
            para_pattern = re.compile(r'([A-Z]+\d+)?\s*(?:paragraph|para)\.?\s+(\d+\.\d+|\d+)', re.IGNORECASE)
            table_pattern = re.compile(r'([A-Z]+\d+)?\s*table\.?\s+(\d+)', re.IGNORECASE)
            section_pattern = re.compile(r'([A-Z]+\d+)?\s*section\.?\s+(\d+\.\d+|\d+)', re.IGNORECASE)
            
            # Process each corpus entry to find references
            for idx, row in enhanced_df.iterrows():
                # Skip processing if no corpus text
                if not isinstance(row['corpus'], str):
                    continue
                    
                corpus_text = row['corpus']
                hierarchy_parts = row['hierarchy'].split(' > ')
                if not hierarchy_parts:
                    continue
                    
                filename = hierarchy_parts[0].split(' - ')[0].strip()
                
                # Check for potential references using regex
                has_potential_refs = any(
                    pattern.search(corpus_text)
                    for pattern in [para_pattern, table_pattern, section_pattern]
                )
                
                references = set()
                
                # If regex found potential references and LM is available, use it
                if has_potential_refs and use_lm:
                    lm_refs = self._extract_references_with_lm(corpus_text, self.generator)
                    references.update(lm_refs)
                
                # If no LM refs found or LM unavailable, use regex
                if not references:
                    # Look for paragraph references
                    para_refs = para_pattern.finditer(corpus_text)
                    for match in para_refs:
                        doc_id, para_num = match.groups()
                        doc_id = doc_id or filename  # Use current doc if not specified
                        references.add(f"{doc_id}, Para {para_num}")
                    
                    # Look for table references
                    table_refs = table_pattern.finditer(corpus_text)
                    for match in table_refs:
                        doc_id, table_num = match.groups()
                        doc_id = doc_id or filename
                        references.add(f"{doc_id}, Table {table_num}")
                    
                    # Look for section references
                    section_refs = section_pattern.finditer(corpus_text)
                    for match in section_refs:
                        doc_id, section_num = match.groups()
                        doc_id = doc_id or filename
                        references.add(f"{doc_id} Section {section_num}")
                
                # Add valid references to reference_additional
                valid_refs = [
                    ref for ref in references
                    if ref in reference_lookup and ref != row['reference']
                ]
                
                if valid_refs:
                    enhanced_df.at[idx, 'reference_additional'] = valid_refs
            
            # Convert lists to comma-separated strings for storage
            enhanced_df['reference_additional'] = enhanced_df['reference_additional'].apply(
                lambda refs: ",".join(refs) if isinstance(refs, list) and refs else ""
            )
            
            # Count references found
            ref_count = (enhanced_df['reference_additional'].str.len() > 0).sum()
            logger.info(f"Found {ref_count} corpus entries with references to other entries")
            
            # Save enhanced knowledge base to new parquet file
            output_path = vectordb_path.replace('.parquet', '_enhanced.parquet')
            enhanced_df.to_parquet(output_path)
            logger.info(f"Saved enhanced knowledge base to {output_path}")
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Failed to extract references: {str(e)}")
            raise RuntimeError(f"Reference extraction failed: {str(e)}")



class GraphBuilder:
    """
    Builds graph representations of the knowledge base using NetworkX.
    Supports both standard graph and hypergraph formats for different query types.
    
    Methods:
        build_standard_graph: Builds a standard directed multigraph from the knowledge base
        build_hypergraph: Builds a hypergraph from the knowledge base
        save_graph_db: Saves the graph to a file in the db folder
        load_graph_db: Loads a graph from a file in the db folder
        merge_graph_dbs: Merges multiple graph databases into one
        visualize_graph: Generates visual representations of the graphs
    """

    def __init__(self, vectordb_file: str = None, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize a GraphBuilder instance.
        
        Args:
            vectordb_file: Path to the knowledge base parquet file
            db_path (Optional[Union[str, Path]]): Root path for the database directory.
                If provided, the db directory will be created at db_root_path/db.
                If None, defaults to Path.cwd() / "db".
        """
        try:
            self.nx = nx
            self.db_dir = Path(db_path) if db_path else Path.cwd() / "db"
            self.db_dir.mkdir(exist_ok=True, parents=True)
            
            # Store the base vectordb file path
            if vectordb_file:
                self.vectordb_path = self.db_dir / vectordb_file
                self.markdown_name = Path(vectordb_file).stem
                if self.markdown_name.startswith('vector_'):
                    self.markdown_name = self.markdown_name[7:]  # Remove 'vector_' prefix
                logger.debug(f"GraphBuilder using vectordb file as a base: {self.vectordb_path}")
            else:
                self.vectordb_path = None
                self.markdown_name = None
                logger.debug("GraphBuilder initialized without a base vectordb file")
                
            self.graph = None  # Standard graph
            self.hypergraph = None  # Hypergraph
            self.semantic_hypergraph = None  # Semantic hypergraph
            self.multilayer_hypergraph = None  # Multilayer hypergraph
            logger.debug("GraphBuilder initialized successfully")
        except ImportError:
            logger.error("NetworkX not found. Please install with: pip install networkx")
            raise
            
    def lookup_node(self, node_reference: str) -> dict:
        """Look up a specific node by its reference from the knowledge base.
        
        Args:
            node_reference: Reference string to look up (e.g., 'APS113 Para 3')
        
        Returns:
            Dictionary with paragraph information if found, None otherwise
            
        Raises:
            FileNotFoundError: If knowledge base file doesn't exist
            ValueError: If paragraph reference not found
        """
        if not self.vectordb_path or not os.path.exists(self.vectordb_path):
            error_msg = f"Knowledge base not found at {self.vectordb_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Load the knowledge base
            df = pd.read_parquet(self.vectordb_path)
            
            # Find the row with the matching reference
            matching_rows = df[df['reference'] == node_reference]
            
            if matching_rows.empty:
                error_msg = f"No node found with reference: {node_reference}"
                logger.warning(error_msg)
                
                # List available references for easier debugging
                available_refs = df['reference'].tolist()
                logger.debug(f"Available references: {available_refs}")
                
                raise ValueError(error_msg)
            
            # Get the first matching row
            row = matching_rows.iloc[0]
            
            # Create a result dictionary with relevant information
            result = {}
            
            # Include all basic columns
            for col in ['reference', 'hierarchy', 'heading', 'corpus']:
                if col in row:
                    result[col] = row[col]
            
            # Include other columns if present
            optional_cols = ['level', 'content_type', 'source_file']
            for col in optional_cols:
                if col in row:
                    result[col] = row[col]
            
            # Don't include vector embeddings by default as they're large
            # But indicate their presence
            vector_cols = [col for col in row.index if 'vector' in col or 'embedding' in col]
            if vector_cols:
                result['has_vectors'] = True
                result['vector_columns'] = vector_cols
            
            logger.info(f"Successfully retrieved node: {node_reference}")
            return result
            
        except Exception as e:
            logger.error(f"Error looking up node: {str(e)}")
            raise
            
    def list_nodes(self) -> list:
        """List all node references available in the knowledge base.
        
        Returns:
            List of node references
            
        Raises:
            FileNotFoundError: If knowledge base file doesn't exist
        """
        if not self.vectordb_path or not os.path.exists(self.vectordb_path):
            error_msg = f"Knowledge base not found at {self.vectordb_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Load the knowledge base
            df = pd.read_parquet(self.vectordb_path)
            
            # Return list of references
            references = df['reference'].tolist()
            logger.info(f"Found {len(references)} nodes")
            return references
            
        except Exception as e:
            logger.error(f"Error listing nodes: {str(e)}")
            raise
    
    def build_graph(self, enhanced_df: Optional[pd.DataFrame] = None) -> nx.MultiDiGraph:
        logger.debug(f"Starting to build standard graph, using {'provided DataFrame' if enhanced_df is not None else 'DataFrame from disk'}")
        start_time = time.time()
        try:
            if enhanced_df is None and self.vectordb_path:
                enhanced_df = pd.read_parquet(self.vectordb_path)
            
            # Create a new directed graph
            G = self.nx.MultiDiGraph()
            
            # Process each row in the DataFrame
            for _, row in enhanced_df.iterrows():
                # Extract document and section info
                hierarchy_parts = row['hierarchy'].split(' > ')
                if not hierarchy_parts:
                    continue
                    
                doc_id = hierarchy_parts[0].split(' - ')[0].strip()
                section_id = row['reference']
                content_id = f"content_{section_id}"
                
                # Add nodes
                G.add_node(doc_id, type='document')
                G.add_node(section_id, type='section', 
                          heading=row.get('heading', ''),
                          level=row.get('level', 0))
                G.add_node(content_id, type='content', 
                          text=row.get('corpus', ''),
                          embedding=row.get('embedding', None))
                
                # Add basic edges
                G.add_edge(doc_id, section_id, type='CONTAINS')
                G.add_edge(section_id, doc_id, type='PART_OF')
                G.add_edge(section_id, content_id, type='HAS_CONTENT')
                
                # Add reference edges
                if row.get('reference_additional'):
                    refs = row['reference_additional'].split(',')
                    for ref in refs:
                        ref = ref.strip()
                        if ref:
                            G.add_edge(section_id, ref, type='REFERENCES')
                
                # Add similarity edges if embeddings exist
                if 'embedding' in row:
                    similar_sections = self._find_similar_sections(row, enhanced_df)
                    for similar_id, similarity in similar_sections:
                        if similar_id != section_id:
                            G.add_edge(section_id, similar_id, 
                                      type='SIMILAR_TO', weight=similarity)
            
            self.graph = G
            logger.info(f"Built standard graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges in {time.time() - start_time:.2f} seconds")
            logger.debug("Finished building standard graph")
            return G
            
        except Exception as e:
            logger.error(f"Failed to build standard graph: {str(e)}")
            raise

    def _find_similar_nodes(self, row: pd.Series, df: pd.DataFrame, 
                             threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find nodes with similar embeddings."""
        if 'embedding' not in row or 'embedding' not in df.columns:
            return []
            
        current_embedding = np.array(row['embedding'])
        similar_sections = []
        
        for _, other_row in df.iterrows():
            if other_row['reference'] != row['reference'] and 'embedding' in other_row:
                other_embedding = np.array(other_row['embedding'])
                similarity = cosine_similarity([current_embedding], [other_embedding])[0][0]
                if similarity >= threshold:
                    similar_sections.append((other_row['reference'], similarity))
        
        return similar_sections

    def build_hypergraph(self, enhanced_df: Optional[pd.DataFrame] = None) -> nx.Graph:
        """Build a hypergraph representation using NetworkX.
        
        Hyperedges connect multiple nodes and represent:
        1. Document groups (sections in same document)
        2. Topic clusters (sections with similar content)
        3. Reference chains (connected references)
        4. Hierarchical levels (sections at same level)
        
        Args:
            enhanced_df: DataFrame with reference_additional column. If None, load from path.
            
        Returns:
            nx.Graph: Bipartite graph representing the hypergraph
        """
        logger.debug(f"Starting to build hypergraph, using {'provided DataFrame' if enhanced_df is not None else 'DataFrame from disk'}")
        start_time = time.time()
        try:
            if enhanced_df is None and self.knowledge_base_path:
                enhanced_df = pd.read_parquet(self.knowledge_base_path)
            
            # Create a bipartite graph to represent the hypergraph
            # One set of nodes are the actual nodes, the other set are hyperedges
            H = self.nx.Graph()
            
            # Track hyperedge IDs
            next_edge_id = 0
            
            # Create document group hyperedges
            doc_groups = enhanced_df.groupby(enhanced_df['hierarchy'].str.split(' > ').str[0])
            for doc_id, group in doc_groups:
                edge_id = f"he_doc_{next_edge_id}"
                next_edge_id += 1
                H.add_node(edge_id, type='hyperedge', edge_type='document_group')
                
                # Connect all sections in this document to the hyperedge
                for _, row in group.iterrows():
                    H.add_node(row['reference'], type='section',
                             heading=row.get('heading', ''),
                             level=row.get('level', 0))
                    H.add_edge(edge_id, row['reference'])
            
            # Create topic cluster hyperedges
            if 'embedding' in enhanced_df.columns:
                clusters = self._cluster_by_embedding(enhanced_df)
                for cluster_idx, cluster_sections in enumerate(clusters):
                    if len(cluster_sections) > 1:  # Only create edges for actual clusters
                        edge_id = f"he_topic_{next_edge_id}"
                        next_edge_id += 1
                        H.add_node(edge_id, type='hyperedge', edge_type='topic_cluster')
                        
                        for section_id in cluster_sections:
                            H.add_edge(edge_id, section_id)
            
            # Create reference chain hyperedges
            reference_chains = self._find_reference_chains(enhanced_df)
            for chain_idx, chain in enumerate(reference_chains):
                if len(chain) > 1:  # Only create edges for actual chains
                    edge_id = f"he_refchain_{next_edge_id}"
                    next_edge_id += 1
                    H.add_node(edge_id, type='hyperedge', edge_type='reference_chain')
                    
                    for section_id in chain:
                        H.add_edge(edge_id, section_id)
            
            # Create level-based hyperedges
            level_groups = enhanced_df.groupby('level')
            for level, group in level_groups:
                edge_id = f"he_level_{next_edge_id}"
                next_edge_id += 1
                H.add_node(edge_id, type='hyperedge', 
                          edge_type='hierarchy_level', level=level)
                
                for _, row in group.iterrows():
                    H.add_edge(edge_id, row['reference'])
            
            self.hypergraph = H
            logger.info(f"Built hypergraph with {H.number_of_nodes()} nodes")
            return H
            
        except Exception as e:
            logger.error(f"Failed to build hypergraph: {str(e)}")
            raise

    def _cluster_by_embedding(self, df: pd.DataFrame, n_clusters: int = 10) -> List[List[str]]:
        """Cluster sections by their embeddings using KMeans."""
        from sklearn.cluster import KMeans
        
        if 'embedding' not in df.columns:
            return []
            
        # Stack embeddings into a matrix
        embeddings = np.vstack(df['embedding'].values)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(df)))
        clusters = kmeans.fit_predict(embeddings)
        
        # Group sections by cluster
        cluster_groups = [[] for _ in range(max(clusters) + 1)]
        for idx, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(df.iloc[idx]['reference'])
            
        return cluster_groups

    def _find_reference_chains(self, df: pd.DataFrame) -> List[List[str]]:
        """Find chains of connected references."""
        chains = []
        visited = set()
        
        def dfs(node: str, current_chain: List[str]):
            visited.add(node)
            current_chain.append(node)
            
            # Get references from this node
            row = df[df['reference'] == node].iloc[0]
            if row.get('reference_additional'):
                refs = row['reference_additional'].split(',')
                for ref in refs:
                    ref = ref.strip()
                    if ref and ref not in visited:
                        dfs(ref, current_chain)
        
        # Start DFS from each unvisited node
        for _, row in df.iterrows():
            if row['reference'] not in visited:
                current_chain = []
                dfs(row['reference'], current_chain)
                if len(current_chain) > 1:  # Only keep chains with multiple nodes
                    chains.append(current_chain)
        
        return chains

    def save_graphdb(self, graph_type: str = 'standard', custom_name: str = None) -> str:
        """Save the graph database to a file in the db folder.
        
        Args:
            graph_type: Type of graph to save ('standard', 'hypergraph', 'semantic', 'multilayer')
            custom_name: Optional custom name for the graph file
            
        Returns:
            str: Path to the saved graph file
            
        Raises:
            ValueError: If the specified graph type is not built yet
        """
        logger.debug(f"Starting to save {graph_type} graph database")
        start_time = time.time()
        
        try:
            # Determine which graph to save
            if graph_type == 'standard':
                graph = self.graph
            elif graph_type == 'hypergraph':
                graph = self.hypergraph
            elif graph_type == 'semantic':
                graph = self.semantic_hypergraph
            elif graph_type == 'multilayer':
                graph = self.multilayer_hypergraph
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
                
            # Check if graph exists
            if graph is None:
                raise ValueError(f"{graph_type.capitalize()} graph not built yet")
            
            # Determine filename
            if custom_name:
                filename = f"graph_{custom_name}.pkl"
            elif self.markdown_name:
                filename = f"graph_{self.markdown_name}.pkl"
            else:
                # Generate a timestamp-based name if no markdown name is available
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"graph_{graph_type}_{timestamp}.pkl"
            
            # Create full path
            filepath = self.db_dir / filename
            
            # Save graph using NetworkX's pickle functionality
            with open(filepath, 'wb') as f:
                pickle.dump(graph, f)
                
            logger.info(f"Saved {graph_type} graph to {filepath} in {time.time() - start_time:.2f} seconds")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save {graph_type} graph: {str(e)}")
            logger.debug(f"Save error details: {traceback.format_exc()}")
            raise
    
    def load_graphdb(self, filepath: str = None, graph_type: str = 'standard') -> nx.Graph:
        """Load a graph database from a file.
        
        Args:
            filepath: Path to the graph file (.pkl or .parquet)
            graph_type: Type of graph to load into ('standard', 'hypergraph', 'semantic', 'multilayer')
            
        Returns:
            nx.Graph: The loaded graph
            
        Raises:
            FileNotFoundError: If the graph file doesn't exist
            ValueError: If the file format is not supported
        """
        logger.debug(f"Starting to load graph database from {filepath}")
        start_time = time.time()
        
        try:
            # If no filepath is provided, try to construct one based on markdown_name
            if filepath is None:
                if self.markdown_name is None:
                    raise ValueError("No filepath provided and no markdown name available")
                filepath = self.db_dir / f"graph_{self.markdown_name}.pkl"
            else:
                # Convert to Path object if string
                filepath = Path(filepath)
                
            # Check if file exists
            if not filepath.exists():
                raise FileNotFoundError(f"Graph file not found: {filepath}")
                
            # Load graph based on file extension
            if filepath.suffix.lower() == '.pkl':
                with open(filepath, 'rb') as f:
                    loaded_graph = pickle.load(f)
            elif filepath.suffix.lower() == '.parquet':
                # For parquet files, we need to rebuild the graph from the data
                df = pd.read_parquet(filepath)
                if graph_type == 'standard':
                    loaded_graph = self.build_standard_graph(df)
                elif graph_type == 'hypergraph':
                    loaded_graph = self.build_hypergraph(df)
                else:
                    raise ValueError(f"Cannot build {graph_type} graph directly from parquet file")
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
                
            # Store the loaded graph in the appropriate attribute
            if graph_type == 'standard':
                self.graph = loaded_graph
            elif graph_type == 'hypergraph':
                self.hypergraph = loaded_graph
            elif graph_type == 'semantic':
                self.semantic_hypergraph = loaded_graph
            elif graph_type == 'multilayer':
                self.multilayer_hypergraph = loaded_graph
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
                
            logger.info(f"Loaded {graph_type} graph from {filepath} with {loaded_graph.number_of_nodes()} nodes and {loaded_graph.number_of_edges()} edges in {time.time() - start_time:.2f} seconds")
            return loaded_graph
            
        except Exception as e:
            logger.error(f"Failed to load graph: {str(e)}")
            logger.debug(f"Load error details: {traceback.format_exc()}")
            raise
    
    def merge_graphdbs(self, filepaths: List[str], output_name: str = None, graph_type: str = 'standard') -> nx.Graph:
        """Merge multiple graph databases into one.
        
        Args:
            filepaths: List of paths to graph files to merge
            output_name: Name for the merged graph file (without extension)
            graph_type: Type of graphs to merge ('standard', 'hypergraph')
            
        Returns:
            nx.Graph: The merged graph
            
        Raises:
            ValueError: If no filepaths are provided or graph types are incompatible
        """
        logger.debug(f"Starting to merge {len(filepaths)} graph databases")
        start_time = time.time()
        
        if not filepaths:
            raise ValueError("No filepaths provided for merging")
            
        try:
            # Initialize merged graph based on type
            if graph_type == 'standard':
                merged_graph = self.nx.MultiDiGraph()
            elif graph_type == 'hypergraph':
                merged_graph = self.nx.Graph()
            else:
                raise ValueError(f"Unsupported graph type for merging: {graph_type}")
                
            # Load and merge each graph
            for i, filepath in enumerate(filepaths):
                logger.debug(f"Loading graph {i+1}/{len(filepaths)}: {filepath}")
                
                # Load the graph without storing it in class attributes
                filepath = Path(filepath)
                if not filepath.exists():
                    logger.warning(f"Graph file not found, skipping: {filepath}")
                    continue
                    
                # Load graph based on file extension
                if filepath.suffix.lower() == '.pkl':
                    with open(filepath, 'rb') as f:
                        graph = pickle.load(f)
                else:
                    logger.warning(f"Unsupported file format, skipping: {filepath.suffix}")
                    continue
                    
                # Check if graph type matches
                if (graph_type == 'standard' and not isinstance(graph, self.nx.MultiDiGraph)) or \
                   (graph_type == 'hypergraph' and not isinstance(graph, self.nx.Graph)):
                    logger.warning(f"Graph type mismatch, skipping: {filepath}")
                    continue
                    
                # Add prefix to node IDs to avoid conflicts
                prefix = f"g{i}_"
                node_mapping = {node: f"{prefix}{node}" if not isinstance(node, str) or not node.startswith('he_') else node 
                              for node in graph.nodes()}
                
                # Create a copy of the graph with renamed nodes
                renamed_graph = self.nx.relabel_nodes(graph, node_mapping)
                
                # Merge into the main graph
                merged_graph.add_nodes_from(renamed_graph.nodes(data=True))
                merged_graph.add_edges_from(renamed_graph.edges(data=True))
                
                logger.debug(f"Merged graph {i+1}: added {renamed_graph.number_of_nodes()} nodes and {renamed_graph.number_of_edges()} edges")
            
            # Generate output name if not provided
            if not output_name:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"merged_{graph_type}_{timestamp}"
                
            # Remove .pkl extension if included
            if output_name.endswith('.pkl'):
                output_name = output_name[:-4]
                
            # Save the merged graph
            filepath = self.db_dir / f"{output_name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(merged_graph, f)
                
            # Store the merged graph in the appropriate attribute
            if graph_type == 'standard':
                self.graph = merged_graph
            elif graph_type == 'hypergraph':
                self.hypergraph = merged_graph
                
            logger.info(f"Merged {len(filepaths)} graphs into {filepath} with {merged_graph.number_of_nodes()} nodes and {merged_graph.number_of_edges()} edges in {time.time() - start_time:.2f} seconds")
            return merged_graph
            
        except Exception as e:
            logger.error(f"Failed to merge graphs: {str(e)}")
            logger.debug(f"Merge error details: {traceback.format_exc()}")
            raise
    
    def view_graphdb(self, graph_type: str = 'standard', output_path: str = None, figsize: tuple = (12, 8), seed: int = 42) -> str:
        """Generate a visual representation of the graph.
        
        Args:
            graph_type: Type of graph to visualize ('standard' or 'hypergraph')
            output_path: Path to save the visualization image. If None, will create
                         a graph_visualizations directory in the current path.
            figsize: Figure size as (width, height) in inches
            seed: Random seed for layout reproducibility
            
        Returns:
            Path to the saved visualization image
            
        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If graph_type is invalid or graph has not been built
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not installed. Please install with: pip install matplotlib")
            raise ImportError("Matplotlib is required for visualization. Install with: pip install matplotlib")
            
        # Validate graph type
        if graph_type not in ['standard', 'hypergraph']:
            raise ValueError("graph_type must be 'standard' or 'hypergraph'")
            
        # Build the graph if it hasn't been built yet
        if graph_type == 'standard':
            if self.graph is None:
                logger.info("Building standard graph first...")
                self.build_standard_graph()
            graph = self.graph
            if graph is None or len(graph.nodes) == 0:
                raise ValueError("Standard graph has not been built or has no nodes")
        else:  # hypergraph
            if self.hypergraph is None:
                logger.info("Building hypergraph first...")
                self.build_hypergraph()
            graph = self.hypergraph
            if graph is None or len(graph.nodes) == 0:
                raise ValueError("Hypergraph has not been built or has no nodes")
                
        # Set up output directory
        if output_path is None:
            viz_dir = os.path.join(os.getcwd(), "graph_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            output_path = os.path.join(viz_dir, f"{graph_type}_graph.png")
            
        # Create visualization
        logger.info(f"Generating {graph_type} graph visualization with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        plt.figure(figsize=figsize)
        pos = self.nx.spring_layout(graph, seed=seed)  # Position nodes using spring layout
        
        try:
            if graph_type == 'standard':
                # Categorize nodes by type for standard graph
                content_nodes = [n for n in graph.nodes if str(n).startswith('content_')]
                reference_nodes = [n for n in graph.nodes if isinstance(n, str) and 'Para' in str(n)]
                hierarchy_nodes = [n for n in graph.nodes if n not in content_nodes and n not in reference_nodes]
                
                # Draw nodes with different colors
                self.nx.draw_networkx_nodes(graph, pos, nodelist=content_nodes, node_color='lightblue', 
                                      node_size=500, alpha=0.8, label="Content")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=reference_nodes, node_color='lightgreen', 
                                      node_size=500, alpha=0.8, label="References")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=hierarchy_nodes, node_color='salmon', 
                                      node_size=700, alpha=0.8, label="Hierarchy")
                
                plt.title("Standard Graph Structure")
                
            else:  # hypergraph
                # Categorize nodes by type for hypergraph
                he_nodes = [n for n in graph.nodes if isinstance(n, str) and str(n).startswith('he_')]
                reference_nodes = [n for n in graph.nodes if isinstance(n, str) and 'Para' in str(n)]
                other_nodes = [n for n in graph.nodes if n not in he_nodes and n not in reference_nodes]
                
                # Draw nodes with different colors
                self.nx.draw_networkx_nodes(graph, pos, nodelist=he_nodes, node_color='purple', 
                                      node_size=700, alpha=0.8, label="Hyperedges")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=reference_nodes, node_color='lightgreen', 
                                      node_size=500, alpha=0.8, label="References")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=other_nodes, node_color='orange', 
                                      node_size=500, alpha=0.8, label="Content")
                
                plt.title("Hypergraph Structure")
            
            # Draw edges and labels for both graph types
            self.nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
            
            # Draw labels with smaller font size to avoid overlap
            self.nx.draw_networkx_labels(graph, pos, font_size=8)
            
            # Add legend and finalize
            plt.legend(loc='upper right', scatterpoints=1)
            plt.axis('off')  # Turn off axis
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to: {output_path}")
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating graph visualization: {str(e)}")
            plt.close()
            raise