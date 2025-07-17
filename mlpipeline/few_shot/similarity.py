"""Similarity engine for few-shot example selection."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .examples import Example, ExampleStore


class SimilarityEngine:
    """Handles similarity-based example selection and augmentation."""
    
    def __init__(
        self,
        example_store: ExampleStore,
        embedding_model: Optional[str] = None,
        use_tfidf_fallback: bool = True
    ):
        """Initialize similarity engine.
        
        Args:
            example_store: ExampleStore instance
            embedding_model: Optional embedding model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
            use_tfidf_fallback: Whether to use TF-IDF as fallback when embeddings unavailable
        """
        self.example_store = example_store
        self.embedding_model = embedding_model
        self.use_tfidf_fallback = use_tfidf_fallback
        self._embedding_encoder = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._examples_cache = []
        
        if embedding_model:
            self._init_embedding_model()
    
    def _init_embedding_model(self) -> None:
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_encoder = SentenceTransformer(self.embedding_model)
        except ImportError:
            print("Warning: sentence-transformers not installed. Using TF-IDF fallback.")
            self._embedding_encoder = None
    
    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if self._embedding_encoder is not None:
            embeddings = self._embedding_encoder.encode(texts)
            return embeddings.tolist()
        else:
            raise RuntimeError("No embedding model available")
    
    def update_example_embeddings(self, force_update: bool = False) -> int:
        """Update embeddings for all examples in the store.
        
        Args:
            force_update: Whether to update existing embeddings
            
        Returns:
            Number of examples updated
        """
        if self._embedding_encoder is None:
            raise RuntimeError("No embedding model available")
        
        examples = self.example_store.list_examples()
        texts_to_encode = []
        examples_to_update = []
        
        for example in examples:
            if force_update or example.embedding is None:
                texts_to_encode.append(example.input_text)
                examples_to_update.append(example)
        
        if not texts_to_encode:
            return 0
        
        embeddings = self.compute_embeddings(texts_to_encode)
        
        count = 0
        for example, embedding in zip(examples_to_update, embeddings):
            if self.example_store.update_embedding(example.id, embedding):
                count += 1
        
        return count
    
    def find_similar_examples(
        self,
        query_text: str,
        k: int = 5,
        similarity_threshold: float = 0.0,
        tags: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Tuple[Example, float]]:
        """Find similar examples to a query text.
        
        Args:
            query_text: Query text to find similar examples for
            k: Number of similar examples to return
            similarity_threshold: Minimum similarity score
            tags: Optional tags to filter examples
            exclude_ids: Optional example IDs to exclude
            
        Returns:
            List of (example, similarity_score) tuples
        """
        # Try embedding-based similarity first
        if self._embedding_encoder is not None:
            return self._find_similar_by_embeddings(
                query_text, k, similarity_threshold, tags, exclude_ids
            )
        elif self.use_tfidf_fallback:
            return self._find_similar_by_tfidf(
                query_text, k, similarity_threshold, tags, exclude_ids
            )
        else:
            raise RuntimeError("No similarity method available")
    
    def _find_similar_by_embeddings(
        self,
        query_text: str,
        k: int,
        similarity_threshold: float,
        tags: Optional[List[str]],
        exclude_ids: Optional[List[str]]
    ) -> List[Tuple[Example, float]]:
        """Find similar examples using embeddings."""
        # Get query embedding
        query_embedding = self.compute_embeddings([query_text])[0]
        
        # Get examples with embeddings
        examples = self.example_store.get_examples_with_embeddings()
        
        # Filter by tags if specified
        if tags:
            examples = [ex for ex in examples if any(tag in ex.tags for tag in tags)]
        
        # Exclude specified IDs
        if exclude_ids:
            examples = [ex for ex in examples if ex.id not in exclude_ids]
        
        if not examples:
            return []
        
        # Compute similarities
        similarities = []
        for example in examples:
            if example.embedding:
                similarity = self._cosine_similarity(query_embedding, example.embedding)
                if similarity >= similarity_threshold:
                    similarities.append((example, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _find_similar_by_tfidf(
        self,
        query_text: str,
        k: int,
        similarity_threshold: float,
        tags: Optional[List[str]],
        exclude_ids: Optional[List[str]]
    ) -> List[Tuple[Example, float]]:
        """Find similar examples using TF-IDF."""
        # Get examples
        examples = self.example_store.list_examples(tags=tags)
        
        # Exclude specified IDs
        if exclude_ids:
            examples = [ex for ex in examples if ex.id not in exclude_ids]
        
        if not examples:
            return []
        
        # Prepare texts
        example_texts = [ex.input_text for ex in examples]
        all_texts = example_texts + [query_text]
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compute similarities
        query_vector = tfidf_matrix[-1]  # Last vector is the query
        example_vectors = tfidf_matrix[:-1]  # All but last are examples
        
        similarities = cosine_similarity(query_vector, example_vectors).flatten()
        
        # Filter and sort
        results = []
        for i, (example, similarity) in enumerate(zip(examples, similarities)):
            if similarity >= similarity_threshold:
                results.append((example, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def select_diverse_examples(
        self,
        examples: List[Example],
        k: int,
        diversity_threshold: float = 0.8
    ) -> List[Example]:
        """Select diverse examples to avoid redundancy.
        
        Args:
            examples: List of candidate examples
            k: Number of examples to select
            diversity_threshold: Minimum diversity score (1 - similarity)
            
        Returns:
            List of selected diverse examples
        """
        if len(examples) <= k:
            return examples
        
        if not examples[0].embedding and self._embedding_encoder:
            # Compute embeddings if not available
            texts = [ex.input_text for ex in examples]
            embeddings = self.compute_embeddings(texts)
            for ex, emb in zip(examples, embeddings):
                ex.embedding = emb
        
        selected = [examples[0]]  # Start with first example
        remaining = examples[1:]
        
        while len(selected) < k and remaining:
            best_candidate = None
            best_diversity = -1
            
            for candidate in remaining:
                if candidate.embedding is None:
                    continue
                
                # Compute minimum similarity to selected examples
                min_similarity = float('inf')
                for selected_ex in selected:
                    if selected_ex.embedding is not None:
                        similarity = self._cosine_similarity(
                            candidate.embedding, selected_ex.embedding
                        )
                        min_similarity = min(min_similarity, similarity)
                
                diversity = 1 - min_similarity
                if diversity > best_diversity and diversity >= (1 - diversity_threshold):
                    best_diversity = diversity
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # If no diverse candidate found, add random one
                selected.append(random.choice(remaining))
                remaining.remove(selected[-1])
        
        return selected
    
    def augment_examples(
        self,
        examples: List[Example],
        augmentation_methods: List[str] = None
    ) -> List[Example]:
        """Generate augmented examples using various techniques.
        
        Args:
            examples: Original examples to augment
            augmentation_methods: List of augmentation methods to apply
            
        Returns:
            List of augmented examples
        """
        if augmentation_methods is None:
            augmentation_methods = ['paraphrase', 'synonym_replacement']
        
        augmented = []
        
        for example in examples:
            for method in augmentation_methods:
                if method == 'paraphrase':
                    aug_example = self._paraphrase_example(example)
                elif method == 'synonym_replacement':
                    aug_example = self._synonym_replacement_example(example)
                elif method == 'back_translation':
                    aug_example = self._back_translation_example(example)
                else:
                    continue
                
                if aug_example:
                    augmented.append(aug_example)
        
        return augmented
    
    def _paraphrase_example(self, example: Example) -> Optional[Example]:
        """Generate paraphrased version of an example."""
        # Placeholder implementation - would need actual paraphrasing model
        # For now, just return a slightly modified version
        paraphrased_input = f"In other words: {example.input_text}"
        
        return Example(
            input_text=paraphrased_input,
            output_text=example.output_text,
            metadata={**example.metadata, 'augmentation': 'paraphrase', 'original_id': example.id},
            tags=example.tags + ['augmented']
        )
    
    def _synonym_replacement_example(self, example: Example) -> Optional[Example]:
        """Generate example with synonym replacement."""
        # Placeholder implementation - would need actual synonym replacement
        # For now, just return a slightly modified version
        words = example.input_text.split()
        if len(words) > 3:
            # Replace a random word (simplified)
            import random
            idx = random.randint(1, len(words) - 2)
            words[idx] = f"[{words[idx]}]"  # Mark as replaced
            
            modified_input = " ".join(words)
            
            return Example(
                input_text=modified_input,
                output_text=example.output_text,
                metadata={**example.metadata, 'augmentation': 'synonym_replacement', 'original_id': example.id},
                tags=example.tags + ['augmented']
            )
        
        return None
    
    def _back_translation_example(self, example: Example) -> Optional[Example]:
        """Generate example using back-translation."""
        # Placeholder implementation - would need translation models
        # For now, just return a slightly modified version
        back_translated_input = f"Translated: {example.input_text}"
        
        return Example(
            input_text=back_translated_input,
            output_text=example.output_text,
            metadata={**example.metadata, 'augmentation': 'back_translation', 'original_id': example.id},
            tags=example.tags + ['augmented']
        )
    
    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about examples in the store.
        
        Returns:
            Dictionary with various statistics
        """
        examples = self.example_store.list_examples()
        
        stats = {
            'total_examples': len(examples),
            'examples_with_embeddings': sum(1 for ex in examples if ex.embedding is not None),
            'unique_tags': len(set(tag for ex in examples for tag in ex.tags)),
            'avg_input_length': np.mean([len(ex.input_text) for ex in examples]) if examples else 0,
            'avg_output_length': np.mean([len(ex.output_text) for ex in examples]) if examples else 0
        }
        
        # Tag distribution
        tag_counts = {}
        for example in examples:
            for tag in example.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats['tag_distribution'] = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))
        
        return stats