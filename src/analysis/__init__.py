"""
Static analysis module for fast project understanding
Provides instant project summaries without requiring full indexing
"""
from src.analysis.project_summarizer import ProjectSummarizer
from src.analysis.file_scorer import FileScorer
from src.analysis.tech_detector import TechDetector

__all__ = ["ProjectSummarizer", "FileScorer", "TechDetector"]
