This document breaks down the sequence of interactions in the `dbbuilder.py` module.

```mermaid
sequenceDiagram
    participant Client
    participant TextParser
    participant TextChunker
    participant VectorBuilder
    participant GraphBuilder
    participant Generator
    
    %% Document Processing Flow
    Client->>TextParser: pdf2md_*(pdf_path)
    TextParser->>Generator: get_hf_ocr()
    Generator-->>TextParser: ocr_text
    TextParser-->>Client: markdown_text
    
    %% Text Chunking Flow
    Client->>TextChunker: length_based_chunking(markdown_file)
    alt Hierarchy-based Chunking
        TextChunker->>TextChunker: hierarchy_based_chunking()
    else Length-based Chunking
        TextChunker->>TextChunker: _chunk_by_length()
    end
    TextChunker-->>Client: chunked_dataframe
    
    %% Vector Database Creation
    Client->>VectorBuilder: create_vectordb(markdown_file, df_headings, chunking_method)
    
    VectorBuilder->>TextChunker: hierarchy_based_chunking()
    VectorBuilder->>Generator: get_embeddings(text_chunks)
    Generator-->>VectorBuilder: embeddings
    
    VectorBuilder->>VectorBuilder: _extract_references_with_lm()
    VectorBuilder-->>Client: vector_db_path
    
    %% Graph Database Creation
    Client->>GraphBuilder: build_graph(enhanced_df)
    
    alt Standard Graph
        GraphBuilder->>GraphBuilder: _build_standard_graph()
    else Hypergraph
        GraphBuilder->>GraphBuilder: _build_hypergraph()
        GraphBuilder->>GraphBuilder: _cluster_by_embedding()
        GraphBuilder->>GraphBuilder: _find_reference_chains()
    end
    
    GraphBuilder-->>Client: graph_object
    
    %% Visualization
    Client->>GraphBuilder: view_graphdb(graph_type)
    GraphBuilder->>GraphBuilder: _visualize_graph()
    GraphBuilder-->>Client: visualization_path
    
    %% Database Management
    alt Save Database
        Client->>VectorBuilder: save_vectordb()
        Client->>GraphBuilder: save_graphdb()
    else Load Database
        Client->>VectorBuilder: load_vectordb()
        Client->>GraphBuilder: load_graphdb()
    end
    GraphBuilder-->>Client: graph_db
    %% Error Handling
    alt Error in any operation
        Component->>Component: Log error
        Component-->>Client: Raise appropriate exception
    end
```

## Key Components and Interactions

1. **TextParser**: Handles document parsing and conversion
   - Converts PDFs to markdown using various methods (OCR, markdown conversion)
   - Supports multiple conversion backends (PyMuPDF, OCR, etc.)

2. **TextChunker**: Manages text segmentation
   - Implements length-based and hierarchy-based chunking
   - Preserves document structure during segmentation
   - Configurable chunk sizes and overlaps

3. **VectorBuilder**: Creates and manages vector databases
   - Generates embeddings for text chunks
   - Extracts and resolves references between chunks
   - Handles vector database storage and retrieval

4. **GraphBuilder**: Builds graph representations
   - Creates standard and hypergraph representations
   - Implements node similarity and clustering
   - Generates visualizations of the knowledge graph

## Main Workflows

### Document Processing
1. Parse PDF documents into markdown
2. Clean and preprocess text
3. Handle different document formats and encodings

### Text Chunking
1. Split documents into manageable chunks
2. Preserve document hierarchy and structure
3. Handle edge cases (tables, lists, code blocks)

### Vector Database Creation
1. Generate embeddings for text chunks
2. Extract and resolve cross-references
3. Store and index vectors for efficient retrieval

### Graph Database Creation
1. Build graph structure from processed text
2. Identify relationships between chunks
3. Support different graph types (standard, hypergraph)

## Error Handling
- Validates input documents and parameters
- Handles malformed or unsupported document formats
- Provides detailed error messages and logging
- Implements retry logic for external service calls
