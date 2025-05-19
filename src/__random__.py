"""
APRA Analysis System - A framework for task-based document analysis and response generation.

This package provides tools and utilities for:
- Building and managing knowledge bases
- Task prompt execution and manipulation
- Response generation and manipulation
- Document retrieval and analysis
"""

from .dbbuilder import DataCleanser, VectorBuilder, GraphBuilder
from .evaluator import ResponseEvaluator
from .generator import Generator, MetaGenerator
from .promptlibrary import PromptLibrary
from .promptops import PromptOperator
from .responseops import ResponseOperator
from .retriever import Retriever
from .utility import DataUtility

__version__ = "0.1.0"
__author__ = "S4ASI"

__all__ = [
    "DataCleanser",
    "VectorBuilder",
    "GraphBuilder",
    "ResponseEvaluator",
    "Generator",
    "PromptLibrary",
    "PromptOperator",
    "ResponseOperator",
    "Retriever",
    "DataUtility"
]
