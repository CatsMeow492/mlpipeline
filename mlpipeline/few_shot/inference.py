"""Few-shot inference pipeline with Hugging Face integration."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import requests

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
        Pipeline
    )
    HAS_TRANSFORMERS = True
except ImportError:
    # Create dummy classes for testing
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    Pipeline = None
    HAS_TRANSFORMERS = False

from .examples import Example, ExampleStore
from .prompts import PromptFormat, PromptManager
from .similarity import SimilarityEngine


logger = logging.getLogger(__name__)


class FewShotInferencePipeline:
    """Few-shot learning inference pipeline with model integration."""
    
    def __init__(
        self,
        model_name_or_path: str,
        prompt_manager: PromptManager,
        example_store: ExampleStore,
        similarity_engine: SimilarityEngine,
        device: str = "auto",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Initialize few-shot inference pipeline.
        
        Args:
            model_name_or_path: Hugging Face model name or local path
            prompt_manager: PromptManager instance
            example_store: ExampleStore instance
            similarity_engine: SimilarityEngine instance
            device: Device to run model on
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.model_name_or_path = model_name_or_path
        self.prompt_manager = prompt_manager
        self.example_store = example_store
        self.similarity_engine = similarity_engine
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._is_openai_compatible = False
        
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            if self.model_name_or_path.startswith("http"):
                # OpenAI-compatible API endpoint
                self._is_openai_compatible = True
                logger.info(f"Using OpenAI-compatible API: {self.model_name_or_path}")
            else:
                # Hugging Face model
                if not HAS_TRANSFORMERS:
                    raise ImportError("transformers library is required for Hugging Face models. Install with: pip install transformers")
                
                logger.info(f"Loading Hugging Face model: {self.model_name_or_path}")
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name_or_path,
                    tokenizer=self._tokenizer,
                    device=self.device,
                    return_full_text=False
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def generate(
        self,
        input_text: str,
        prompt_template: str,
        num_examples: int = 3,
        similarity_threshold: float = 0.1,
        example_tags: Optional[List[str]] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using few-shot learning.
        
        Args:
            input_text: Input text to generate response for
            prompt_template: Name of prompt template to use
            num_examples: Number of examples to include
            similarity_threshold: Minimum similarity for example selection
            example_tags: Optional tags to filter examples
            prompt_variables: Additional variables for prompt template
            generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Find similar examples
        similar_examples = self.similarity_engine.find_similar_examples(
            query_text=input_text,
            k=num_examples,
            similarity_threshold=similarity_threshold,
            tags=example_tags
        )
        
        # Prepare examples for prompt
        examples_for_prompt = []
        for example, similarity in similar_examples:
            examples_for_prompt.append({
                'input': example.input_text,
                'output': example.output_text,
                'user': example.input_text,  # For chat format
                'assistant': example.output_text  # For chat format
            })
        
        # Get prompt template
        template = self.prompt_manager.get_template(prompt_template)
        
        # Prepare prompt variables
        variables = {
            'input': input_text,
            'examples': examples_for_prompt,
            **(prompt_variables or {})
        }
        
        # Render prompt
        if template.format == PromptFormat.INSTRUCTION:
            prompt = self.prompt_manager.format_prompt(
                PromptFormat.INSTRUCTION,
                input_text,
                system_message=variables.get('system_message'),
                examples=examples_for_prompt
            )
        elif template.format == PromptFormat.CHAT:
            prompt = self.prompt_manager.format_prompt(
                PromptFormat.CHAT,
                input_text,
                system_message=variables.get('system_message'),
                examples=examples_for_prompt
            )
        elif template.format == PromptFormat.COMPLETION:
            prompt = self.prompt_manager.format_prompt(
                PromptFormat.COMPLETION,
                input_text,
                examples=examples_for_prompt
            )
        else:
            # Use template rendering
            prompt = self.prompt_manager.render_template(
                prompt_template,
                variables
            )
        
        # Generate response
        if self._is_openai_compatible:
            response = self._generate_openai_compatible(prompt, generation_kwargs)
        else:
            response = self._generate_huggingface(prompt, generation_kwargs)
        
        return {
            'generated_text': response,
            'prompt': prompt,
            'examples_used': len(examples_for_prompt),
            'example_similarities': [sim for _, sim in similar_examples],
            'template_name': prompt_template,
            'template_format': template.format.value
        }
    
    def _generate_huggingface(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate using Hugging Face pipeline."""
        kwargs = {
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'do_sample': True,
            'pad_token_id': self._tokenizer.eos_token_id,
            **(generation_kwargs or {})
        }
        
        try:
            outputs = self._pipeline(prompt, **kwargs)
            if outputs and len(outputs) > 0:
                return outputs[0]['generated_text'].strip()
            else:
                return ""
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_openai_compatible(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate using OpenAI-compatible API."""
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Add API key if available
        api_key = generation_kwargs.get('api_key') if generation_kwargs else None
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        generation_kwargs = generation_kwargs or {}
        data = {
            'prompt': prompt,
            'max_tokens': generation_kwargs.get('max_tokens', self.max_length),
            'temperature': generation_kwargs.get('temperature', self.temperature),
            'top_p': generation_kwargs.get('top_p', self.top_p),
            'stop': generation_kwargs.get('stop', []),
        }
        
        try:
            response = requests.post(
                f"{self.model_name_or_path}/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['text'].strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"OpenAI-compatible API call failed: {e}")
            raise
    
    def batch_generate(
        self,
        inputs: List[str],
        prompt_template: str,
        num_examples: int = 3,
        similarity_threshold: float = 0.1,
        example_tags: Optional[List[str]] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple inputs.
        
        Args:
            inputs: List of input texts
            prompt_template: Name of prompt template to use
            num_examples: Number of examples to include
            similarity_threshold: Minimum similarity for example selection
            example_tags: Optional tags to filter examples
            prompt_variables: Additional variables for prompt template
            generation_kwargs: Additional generation parameters
            
        Returns:
            List of generation results
        """
        results = []
        
        for input_text in inputs:
            try:
                result = self.generate(
                    input_text=input_text,
                    prompt_template=prompt_template,
                    num_examples=num_examples,
                    similarity_threshold=similarity_threshold,
                    example_tags=example_tags,
                    prompt_variables=prompt_variables,
                    generation_kwargs=generation_kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for input '{input_text}': {e}")
                results.append({
                    'generated_text': '',
                    'error': str(e),
                    'input': input_text
                })
        
        return results
    
    def evaluate_few_shot_performance(
        self,
        test_examples: List[Example],
        prompt_template: str,
        num_examples: int = 3,
        similarity_threshold: float = 0.1,
        example_tags: Optional[List[str]] = None,
        exclude_test_from_examples: bool = True
    ) -> Dict[str, Any]:
        """Evaluate few-shot performance on test examples.
        
        Args:
            test_examples: Examples to test on
            prompt_template: Name of prompt template to use
            num_examples: Number of examples to include
            similarity_threshold: Minimum similarity for example selection
            example_tags: Optional tags to filter examples
            exclude_test_from_examples: Whether to exclude test examples from few-shot examples
            
        Returns:
            Evaluation results
        """
        results = []
        correct = 0
        total = len(test_examples)
        
        for test_example in test_examples:
            exclude_ids = [test_example.id] if exclude_test_from_examples else None
            
            # Find similar examples (excluding the test example itself)
            similar_examples = self.similarity_engine.find_similar_examples(
                query_text=test_example.input_text,
                k=num_examples,
                similarity_threshold=similarity_threshold,
                tags=example_tags,
                exclude_ids=exclude_ids
            )
            
            # Generate prediction
            try:
                result = self.generate(
                    input_text=test_example.input_text,
                    prompt_template=prompt_template,
                    num_examples=num_examples,
                    similarity_threshold=similarity_threshold,
                    example_tags=example_tags
                )
                
                predicted = result['generated_text']
                expected = test_example.output_text
                
                # Simple exact match evaluation (could be improved)
                is_correct = predicted.strip().lower() == expected.strip().lower()
                if is_correct:
                    correct += 1
                
                results.append({
                    'input': test_example.input_text,
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct,
                    'examples_used': result['examples_used'],
                    'example_similarities': result['example_similarities']
                })
                
            except Exception as e:
                logger.error(f"Evaluation failed for example {test_example.id}: {e}")
                results.append({
                    'input': test_example.input_text,
                    'expected': test_example.output_text,
                    'predicted': '',
                    'correct': False,
                    'error': str(e)
                })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results,
            'prompt_template': prompt_template,
            'num_examples': num_examples,
            'similarity_threshold': similarity_threshold
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        info = {
            'model_name_or_path': self.model_name_or_path,
            'is_openai_compatible': self._is_openai_compatible,
            'device': self.device,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p
        }
        
        if self._tokenizer:
            info.update({
                'vocab_size': self._tokenizer.vocab_size,
                'model_max_length': getattr(self._tokenizer, 'model_max_length', None),
                'pad_token': self._tokenizer.pad_token,
                'eos_token': self._tokenizer.eos_token
            })
        
        return info


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize OpenAI-compatible client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate completion using the API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        data = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop or []
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['text']
        else:
            return ""
    
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate chat completion using the API.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        data = {
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop or []
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return ""