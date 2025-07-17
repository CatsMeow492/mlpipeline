"""Few-shot learning components."""

from .prompts import PromptManager, PromptFormat, PromptTemplate
from .examples import ExampleStore, Example
from .similarity import SimilarityEngine
from .inference import FewShotInferencePipeline, OpenAICompatibleClient

__all__ = [
    "PromptManager", "PromptFormat", "PromptTemplate",
    "ExampleStore", "Example",
    "SimilarityEngine",
    "FewShotInferencePipeline", "OpenAICompatibleClient"
]