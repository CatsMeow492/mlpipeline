"""Tests for few-shot example store and similarity engine."""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from mlpipeline.few_shot import ExampleStore, Example, SimilarityEngine


class TestExample:
    """Test Example model."""
    
    def test_create_example(self):
        """Test creating an example."""
        example = Example(
            input_text="What is the capital of France?",
            output_text="Paris",
            metadata={"category": "geography"},
            tags=["geography", "capitals"]
        )
        
        assert example.input_text == "What is the capital of France?"
        assert example.output_text == "Paris"
        assert example.metadata["category"] == "geography"
        assert "geography" in example.tags


class TestExampleStore:
    """Test ExampleStore functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            yield tmp.name
        Path(tmp.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def example_store(self, temp_db):
        """Create ExampleStore instance."""
        return ExampleStore(db_path=temp_db)
    
    def test_add_example(self, example_store):
        """Test adding an example."""
        example_id = example_store.add_example(
            input_text="Hello world",
            output_text="Greeting",
            metadata={"type": "greeting"},
            tags=["greeting", "simple"]
        )
        
        assert example_id is not None
        assert len(example_id) > 0
    
    def test_get_example(self, example_store):
        """Test retrieving an example."""
        # Add example
        example_id = example_store.add_example(
            input_text="Test input",
            output_text="Test output",
            tags=["test"]
        )
        
        # Retrieve example
        retrieved = example_store.get_example(example_id)
        assert retrieved is not None
        assert retrieved.input_text == "Test input"
        assert retrieved.output_text == "Test output"
        assert "test" in retrieved.tags
    
    def test_get_nonexistent_example(self, example_store):
        """Test retrieving non-existent example."""
        result = example_store.get_example("nonexistent")
        assert result is None
    
    def test_list_examples(self, example_store):
        """Test listing examples."""
        # Add multiple examples
        example_store.add_example("Input 1", "Output 1", tags=["tag1"])
        example_store.add_example("Input 2", "Output 2", tags=["tag2"])
        example_store.add_example("Input 3", "Output 3", tags=["tag1", "tag2"])
        
        # List all examples
        all_examples = example_store.list_examples()
        assert len(all_examples) == 3
        
        # Filter by tags
        tag1_examples = example_store.list_examples(tags=["tag1"])
        assert len(tag1_examples) == 2
        
        # Test pagination
        limited_examples = example_store.list_examples(limit=2)
        assert len(limited_examples) == 2
    
    def test_delete_example(self, example_store):
        """Test deleting an example."""
        # Add example
        example_id = example_store.add_example("To delete", "Delete me")
        
        # Verify it exists
        assert example_store.get_example(example_id) is not None
        
        # Delete it
        success = example_store.delete_example(example_id)
        assert success is True
        
        # Verify it's gone
        assert example_store.get_example(example_id) is None
    
    def test_update_embedding(self, example_store):
        """Test updating example embedding."""
        # Add example
        example_id = example_store.add_example("Test", "Test output")
        
        # Update embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        success = example_store.update_embedding(example_id, embedding)
        assert success is True
        
        # Verify embedding was updated
        retrieved = example_store.get_example(example_id)
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 5
        assert abs(retrieved.embedding[0] - 0.1) < 1e-6
    
    def test_count_examples(self, example_store):
        """Test counting examples."""
        # Initially empty
        assert example_store.count_examples() == 0
        
        # Add examples
        example_store.add_example("Input 1", "Output 1", tags=["tag1"])
        example_store.add_example("Input 2", "Output 2", tags=["tag2"])
        
        # Count all
        assert example_store.count_examples() == 2
        
        # Count with tag filter
        assert example_store.count_examples(tags=["tag1"]) == 1
    
    def test_export_import_examples(self, example_store, tmp_path):
        """Test exporting and importing examples."""
        # Add examples
        example_store.add_example(
            "Question 1", "Answer 1",
            metadata={"category": "test"},
            tags=["export_test"]
        )
        example_store.add_example(
            "Question 2", "Answer 2",
            tags=["export_test"]
        )
        
        # Export to JSON
        export_file = tmp_path / "examples.json"
        example_store.export_examples(export_file, format="json")
        assert export_file.exists()
        
        # Create new store and import
        import_db = tmp_path / "import.db"
        import_store = ExampleStore(db_path=import_db)
        
        count = import_store.import_examples(export_file, format="json")
        assert count == 2
        
        # Verify imported examples
        imported_examples = import_store.list_examples()
        assert len(imported_examples) == 2


class TestSimilarityEngine:
    """Test SimilarityEngine functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            yield tmp.name
        Path(tmp.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def example_store(self, temp_db):
        """Create ExampleStore with sample data."""
        store = ExampleStore(db_path=temp_db)
        
        # Add sample examples
        store.add_example(
            "What is machine learning?",
            "Machine learning is a subset of AI",
            tags=["ml", "definition"]
        )
        store.add_example(
            "How does neural network work?",
            "Neural networks use interconnected nodes",
            tags=["ml", "neural_networks"]
        )
        store.add_example(
            "What is deep learning?",
            "Deep learning uses multi-layer neural networks",
            tags=["ml", "deep_learning"]
        )
        
        return store
    
    @pytest.fixture
    def similarity_engine(self, example_store):
        """Create SimilarityEngine instance."""
        return SimilarityEngine(
            example_store=example_store,
            use_tfidf_fallback=True
        )
    
    def test_find_similar_examples_tfidf(self, similarity_engine):
        """Test finding similar examples using TF-IDF."""
        similar = similarity_engine.find_similar_examples(
            query_text="What is artificial intelligence?",
            k=2,
            similarity_threshold=0.0
        )
        
        assert len(similar) <= 2
        assert all(isinstance(score, float) for _, score in similar)
        assert all(0 <= score <= 1 for _, score in similar)
    
    def test_find_similar_examples_with_tags(self, similarity_engine):
        """Test finding similar examples with tag filtering."""
        similar = similarity_engine.find_similar_examples(
            query_text="neural network architecture",
            k=5,
            tags=["neural_networks"]
        )
        
        # Should only return examples with neural_networks tag
        for example, _ in similar:
            assert "neural_networks" in example.tags
    
    def test_find_similar_examples_exclude_ids(self, similarity_engine):
        """Test excluding specific example IDs."""
        # Get all examples first
        all_examples = similarity_engine.example_store.list_examples()
        exclude_id = all_examples[0].id if all_examples else None
        
        if exclude_id:
            similar = similarity_engine.find_similar_examples(
                query_text="machine learning concepts",
                k=5,
                exclude_ids=[exclude_id]
            )
            
            # Excluded example should not be in results
            result_ids = [ex.id for ex, _ in similar]
            assert exclude_id not in result_ids
    
    def test_select_diverse_examples(self, similarity_engine):
        """Test selecting diverse examples."""
        examples = similarity_engine.example_store.list_examples()
        
        if len(examples) >= 2:
            diverse = similarity_engine.select_diverse_examples(
                examples=examples,
                k=2,
                diversity_threshold=0.5
            )
            
            assert len(diverse) <= 2
            assert all(isinstance(ex, Example) for ex in diverse)
    
    def test_get_example_statistics(self, similarity_engine):
        """Test getting example statistics."""
        stats = similarity_engine.get_example_statistics()
        
        assert 'total_examples' in stats
        assert 'examples_with_embeddings' in stats
        assert 'unique_tags' in stats
        assert 'avg_input_length' in stats
        assert 'avg_output_length' in stats
        assert 'tag_distribution' in stats
        
        assert stats['total_examples'] >= 0
        assert stats['unique_tags'] >= 0
    
    def test_augment_examples(self, similarity_engine):
        """Test example augmentation."""
        examples = similarity_engine.example_store.list_examples()
        
        if examples:
            augmented = similarity_engine.augment_examples(
                examples=examples[:1],
                augmentation_methods=['paraphrase']
            )
            
            assert len(augmented) >= 0
            for aug_ex in augmented:
                assert isinstance(aug_ex, Example)
                assert 'augmented' in aug_ex.tags
                assert 'augmentation' in aug_ex.metadata