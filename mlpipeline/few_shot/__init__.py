"""Few-shot learning components."""

from .prompts import PromptManager
from .examples import ExampleStore
from .similarity import SimilarityEngine

__all__ = ["PromptManager", "ExampleStore", "SimilarityEngine"]