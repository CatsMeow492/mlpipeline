"""Tests for few-shot inference pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mlpipeline.few_shot import (
    FewShotInferencePipeline,
    PromptManager,
    ExampleStore,
    SimilarityEngine,
    PromptFormat,
    OpenAICompatibleClient
)


class TestFewShotInferencePipeline:
    """Test FewShotInferencePipeline functionality."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            yield {
                'templates': tmpdir / 'templates',
                'db': tmpdir / 'examples.db'
            }
    
    @pytest.fixture
    def components(self, temp_dirs):
        """Create pipeline components."""
        # Create prompt manager
        prompt_manager = PromptManager(templates_dir=temp_dirs['templates'])
        prompt_manager.create_template(
            name="classification",
            template="Classify the sentiment: {{ input }}",
            format=PromptFormat.COMPLETION,
            save=False
        )
        
        # Create example store with sample data
        example_store = ExampleStore(db_path=temp_dirs['db'])
        example_store.add_example(
            "I love this product!",
            "Positive",
            tags=["sentiment"]
        )
        example_store.add_example(
            "This is terrible.",
            "Negative",
            tags=["sentiment"]
        )
        
        # Create similarity engine
        similarity_engine = SimilarityEngine(
            example_store=example_store,
            use_tfidf_fallback=True
        )
        
        return {
            'prompt_manager': prompt_manager,
            'example_store': example_store,
            'similarity_engine': similarity_engine
        }
    
    @patch('mlpipeline.few_shot.inference.HAS_TRANSFORMERS', True)
    @patch('mlpipeline.few_shot.inference.pipeline')
    @patch('mlpipeline.few_shot.inference.AutoTokenizer')
    def test_init_huggingface_model(self, mock_tokenizer, mock_pipeline, components):
        """Test initializing with Hugging Face model."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create pipeline
        pipeline = FewShotInferencePipeline(
            model_name_or_path="gpt2",
            **components
        )
        
        assert pipeline.model_name_or_path == "gpt2"
        assert not pipeline._is_openai_compatible
        assert pipeline._pipeline is not None
    
    def test_init_openai_compatible(self, components):
        """Test initializing with OpenAI-compatible API."""
        pipeline = FewShotInferencePipeline(
            model_name_or_path="http://localhost:8000",
            **components
        )
        
        assert pipeline._is_openai_compatible
        assert pipeline._pipeline is None
    
    @patch('mlpipeline.few_shot.inference.HAS_TRANSFORMERS', True)
    @patch('mlpipeline.few_shot.inference.pipeline')
    @patch('mlpipeline.few_shot.inference.AutoTokenizer')
    def test_generate_huggingface(self, mock_tokenizer, mock_pipeline, components):
        """Test generation with Hugging Face model."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'generated_text': 'Positive'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create pipeline
        pipeline = FewShotInferencePipeline(
            model_name_or_path="gpt2",
            **components
        )
        
        # Generate
        result = pipeline.generate(
            input_text="This is amazing!",
            prompt_template="classification",
            num_examples=1
        )
        
        assert 'generated_text' in result
        assert 'prompt' in result
        assert 'examples_used' in result
        assert result['template_name'] == "classification"
    
    @patch('mlpipeline.few_shot.inference.requests.post')
    def test_generate_openai_compatible(self, mock_post, components):
        """Test generation with OpenAI-compatible API."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{'text': 'Positive'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create pipeline
        pipeline = FewShotInferencePipeline(
            model_name_or_path="http://localhost:8000",
            **components
        )
        
        # Generate
        result = pipeline.generate(
            input_text="This is great!",
            prompt_template="classification",
            num_examples=1
        )
        
        assert 'generated_text' in result
        assert result['generated_text'] == 'Positive'
        assert 'prompt' in result
        assert 'examples_used' in result
    
    @patch('mlpipeline.few_shot.inference.HAS_TRANSFORMERS', True)
    @patch('mlpipeline.few_shot.inference.pipeline')
    @patch('mlpipeline.few_shot.inference.AutoTokenizer')
    def test_batch_generate(self, mock_tokenizer, mock_pipeline, components):
        """Test batch generation."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'generated_text': 'Positive'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create pipeline
        pipeline = FewShotInferencePipeline(
            model_name_or_path="gpt2",
            **components
        )
        
        # Batch generate
        inputs = ["Great product!", "Awful experience"]
        results = pipeline.batch_generate(
            inputs=inputs,
            prompt_template="classification",
            num_examples=1
        )
        
        assert len(results) == 2
        assert all('generated_text' in result for result in results)
    
    def test_get_model_info(self, components):
        """Test getting model information."""
        pipeline = FewShotInferencePipeline(
            model_name_or_path="http://localhost:8000",
            **components
        )
        
        info = pipeline.get_model_info()
        
        assert 'model_name_or_path' in info
        assert 'is_openai_compatible' in info
        assert 'device' in info
        assert 'max_length' in info
        assert info['model_name_or_path'] == "http://localhost:8000"
        assert info['is_openai_compatible'] is True


class TestOpenAICompatibleClient:
    """Test OpenAICompatibleClient functionality."""
    
    @pytest.fixture
    def client(self):
        """Create OpenAI-compatible client."""
        return OpenAICompatibleClient(
            base_url="http://localhost:8000",
            api_key="test-key"
        )
    
    @patch('mlpipeline.few_shot.inference.requests.post')
    def test_complete(self, mock_post, client):
        """Test completion API call."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{'text': 'Generated text'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Make completion request
        result = client.complete(
            prompt="Test prompt",
            max_tokens=50,
            temperature=0.7
        )
        
        assert result == "Generated text"
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:8000/completions"
        assert 'Authorization' in call_args[1]['headers']
        assert call_args[1]['headers']['Authorization'] == 'Bearer test-key'
    
    @patch('mlpipeline.few_shot.inference.requests.post')
    def test_chat_complete(self, mock_post, client):
        """Test chat completion API call."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Chat response'}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Make chat completion request
        messages = [
            {'role': 'user', 'content': 'Hello'}
        ]
        result = client.chat_complete(
            messages=messages,
            max_tokens=50
        )
        
        assert result == "Chat response"
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:8000/chat/completions"