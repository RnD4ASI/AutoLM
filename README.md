# AutoLM

## Project Overview
Auto GAI is an advanced, modular framework for automated generation, evaluation, and refinement of AI prompts and knowledge bases. It is designed to support research, experimentation, and deployment of prompt engineering workflows, including retrieval-augmented generation, evaluation, and publishing to external platforms.

## Architecture Diagram

```mermaid
graph TD
    %% Legend/Description Box
    Legend["<b>Legend</b><br><br><b>Solid line</b>: Direct import/dependency<br><b>Dotted line</b>: Indirect/optional/weak dependency<br><b>Double-headed dotted</b>: Circular import risk"]
    style Legend fill:#f9f,stroke:#333,stroke-width:2px
    Legend --- A

    %% Hierarchical Dependency Structure
    subgraph Interface Layer
        A[User Interface]
    end
    subgraph Orchestration Layer
        B[publisher.py]
        F[topologist.py]
        K[planner.py]
        L[executor.py]
    end
    subgraph Core Modules
        D[editor.py]
        C[generator.py]
        E[evaluator.py]
        G[dbbuilder.py]
        H[retriever.py]
    end
    subgraph Core Utilities
        I[utility.py]
        J[logging.py]
    end
    subgraph Optimisation Modules
        M[hh_swe.py]
        N[hh_mle.py]
        O[hh_pm.py]
    end

    %% Direct dependencies (solid lines)
    A --> B
    B --> I
    C --> E
    C --> G
    D --> F
    E --> F
    E --> H
    G --> H
    K --> L
    L --> F
    L --> G
    L --> H
    L --> M
    L --> N
    L --> O
    K --> F
    K --> G
    K --> H

    %% Indirect/optional dependencies (dotted lines)
    A -.-> C
    A -.-> D
    A -.-> E
    A -.-> F
    A -.-> G
    A -.-> H
    C -.-> F
    B -.-> E
    B -.-> F
    B -.-> H
    K -.-> L
    L -.-> E
    L -.-> C

    %% Circular import risk (double-headed dotted)
    H <-.-> G

    %% Styling for subgraphs
    style Interface Layer fill:#e3f2fd,stroke:#2196f3
    style Orchestration Layer fill:#e8f5e9,stroke:#43a047
    style Core Modules fill:#fff3e0,stroke:#fb8c00
    style Core Utilities fill:#f3e5f5,stroke:#8e24aa
    style Optimisation Modules fill:#e0f7fa,stroke:#00838f
```

**Diagram Description:**
- **Solid lines** indicate direct imports/dependencies between modules.
- **Dotted lines** indicate indirect, optional, or weak dependencies.
- **Double-headed dotted lines** highlight circular import risks.
- Modules are grouped by hierarchy: Interface, Orchestration, Core Modules, Core Utilities, and Optimisation Modules.

*For a detailed, interactive dependency diagram and explanations, see `docs/analysis/module_dependency.md`.*

## Component Descriptions

- **logging.py**: Configures logging using loguru for consistent, formatted logs.
- **utility.py**: Provides utility functions and helpers used throughout the project.
- **generator.py**: Handles model interactions and text generation, supporting Azure OpenAI and HuggingFace models.
- **editor.py**: Implements prompt enhancement and template adoption strategies.
- **evaluator.py**: Evaluates generated text using metrics like BLEU and includes retrieval-based evaluation.
- **topologist.py**: Defines and orchestrates prompt execution topologies, including advanced strategies (e.g., genetic algorithms, regenerative majority synthesis).
- **dbbuilder.py**: Manages data cleansing, chunking, and vector database construction.
- **publisher.py**: Publishes content to external platforms (e.g., Confluence) with Markdown and LaTeX support.
- **retriever.py**: Implements retrieval and reranking for vector databases, supporting advanced document search and reranking.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd auto_gai
   ```
2. **Set up a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   - If you have a `requirements.txt`, run:
     ```bash
     pip install -r requirements.txt
     ```
   - Otherwise, install core dependencies manually (example):
     ```bash
     pip install loguru pandas numpy scikit-learn openai
     ```
   - Add any other dependencies as needed for your use case.

## Development Workflow

- **Branching:**
  - Use feature branches for new features (`feature/<name>`), bugfixes (`bugfix/<name>`), and documentation (`docs/<name>`).
- **Code Style:**
  - Follow [PEP8](https://peps.python.org/pep-0008/) for Python code.
  - Use type hints and docstrings for all public functions/classes.
- **Documentation:**
  - Update the `docs/` directory for architectural or analytical changes.
  - Keep `README.md` and module docstrings up to date.
- **Commits:**
  - Write clear, descriptive commit messages.

## Testing Procedures

- **Unit and Integration Tests:**
  - Place tests in a `tests/` directory (create if missing).
  - Use `pytest` or Python’s built-in `unittest` framework.
  - Example to run all tests:
    ```bash
    pytest
    ```
- **Test Coverage:**
  - Aim for high coverage on core modules (generator, evaluator, retriever, dbbuilder, topologist).
- **Continuous Integration:**
  - Integrate with GitHub Actions or another CI tool for automated testing (recommended).

---

For more details, see the [module dependency analysis](docs/analysis/module_dependency.md) and in-code documentation.

---
*Last updated: 2025-05-18*
