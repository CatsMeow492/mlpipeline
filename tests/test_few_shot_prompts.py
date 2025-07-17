"""Tests for few-shot prompt management system."""

import tempfile
from pathlib import Path

import pytest

from mlpipeline.few_shot import PromptManager, PromptFormat, PromptTemplate


class TestPromptTemplate:
    """Test PromptTemplate model."""
    
    def test_create_valid_template(self):
        """Test creating a valid prompt template."""
        template = PromptTemplate(
            name="test_template",
            format=PromptFormat.INSTRUCTION,
            template="Classify the following text: {{ input }}"
        )
        
        assert template.name == "test_template"
        assert template.format == PromptFormat.INSTRUCTION
        assert template.template == "Classify the following text: {{ input }}"
        assert "input" in template.variables
    
    def test_invalid_template_syntax(self):
        """Test validation of invalid template syntax."""
        with pytest.raises(ValueError, match="Invalid template syntax"):
            PromptTemplate(
                name="invalid",
                format=PromptFormat.INSTRUCTION,
                template="Invalid template {{ unclosed"
            )
    
    def test_variable_extraction(self):
        """Test automatic variable extraction."""
        template = PromptTemplate(
            name="multi_var",
            format=PromptFormat.INSTRUCTION,
            template="Process {{ input }} with {{ method }} and {{ params }}"
        )
        
        expected_vars = {"input", "method", "params"}
        assert set(template.variables) == expected_vars


class TestPromptManager:
    """Test PromptManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def prompt_manager(self, temp_dir):
        """Create PromptManager instance."""
        return PromptManager(templates_dir=temp_dir)
    
    def test_create_template(self, prompt_manager):
        """Test creating a new template."""
        template = prompt_manager.create_template(
            name="classification",
            template="Classify: {{ input }}",
            format=PromptFormat.INSTRUCTION,
            description="Simple classification template",
            tags=["classification", "simple"],
            save=False
        )
        
        assert template.name == "classification"
        assert template.format == PromptFormat.INSTRUCTION
        assert "input" in template.variables
        assert "classification" in template.tags
    
    def test_get_template(self, prompt_manager):
        """Test retrieving a template."""
        # Create template
        prompt_manager.create_template(
            name="test",
            template="Test: {{ input }}",
            format=PromptFormat.COMPLETION,
            save=False
        )
        
        # Retrieve template
        retrieved = prompt_manager.get_template("test")
        assert retrieved.name == "test"
        assert retrieved.format == PromptFormat.COMPLETION
    
    def test_get_nonexistent_template(self, prompt_manager):
        """Test retrieving non-existent template."""
        with pytest.raises(KeyError, match="Template 'nonexistent' not found"):
            prompt_manager.get_template("nonexistent")
    
    def test_list_templates(self, prompt_manager):
        """Test listing templates."""
        # Create multiple templates
        prompt_manager.create_template(
            name="template1",
            template="Template 1: {{ input }}",
            format=PromptFormat.INSTRUCTION,
            save=False
        )
        prompt_manager.create_template(
            name="template2",
            template="Template 2: {{ input }}",
            format=PromptFormat.CHAT,
            save=False
        )
        
        # List all templates
        templates = prompt_manager.list_templates()
        assert len(templates) == 2
        
        # Filter by format
        instruction_templates = prompt_manager.list_templates(
            format_filter=PromptFormat.INSTRUCTION
        )
        assert len(instruction_templates) == 1
        assert instruction_templates[0]['format'] == PromptFormat.INSTRUCTION
    
    def test_render_template(self, prompt_manager):
        """Test rendering a template with variables."""
        prompt_manager.create_template(
            name="greeting",
            template="Hello {{ name }}, welcome to {{ place }}!",
            format=PromptFormat.COMPLETION,
            save=False
        )
        
        rendered = prompt_manager.render_template(
            "greeting",
            {"name": "Alice", "place": "Wonderland"}
        )
        
        assert rendered == "Hello Alice, welcome to Wonderland!"
    
    def test_render_template_missing_variables(self, prompt_manager):
        """Test rendering template with missing variables."""
        prompt_manager.create_template(
            name="incomplete",
            template="Hello {{ name }}, you are {{ age }} years old",
            format=PromptFormat.COMPLETION,
            save=False
        )
        
        with pytest.raises(ValueError, match="Missing required variables"):
            prompt_manager.render_template("incomplete", {"name": "Bob"})
    
    def test_validate_template(self, prompt_manager):
        """Test template validation."""
        # Valid template
        result = prompt_manager.validate_template("Hello {{ name }}!")
        assert result['valid'] is True
        assert "name" in result['variables']
        assert result['error'] is None
        
        # Invalid template
        result = prompt_manager.validate_template("Hello {{ unclosed")
        assert result['valid'] is False
        assert result['error'] is not None
    
    def test_format_instruction_prompt(self, prompt_manager):
        """Test formatting instruction prompt."""
        examples = [
            {"input": "Happy news", "output": "Positive"},
            {"input": "Sad story", "output": "Negative"}
        ]
        
        formatted = prompt_manager.format_prompt(
            PromptFormat.INSTRUCTION,
            "Great day!",
            system_message="Classify sentiment",
            examples=examples
        )
        
        assert "System: Classify sentiment" in formatted
        assert "Examples:" in formatted
        assert "Happy news" in formatted
        assert "Positive" in formatted
        assert "Instruction: Great day!" in formatted
    
    def test_format_chat_prompt(self, prompt_manager):
        """Test formatting chat prompt."""
        examples = [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "How are you?", "assistant": "I'm doing well, thanks!"}
        ]
        
        formatted = prompt_manager.format_prompt(
            PromptFormat.CHAT,
            "What's the weather?",
            system_message="You are a helpful assistant",
            examples=examples
        )
        
        assert "System: You are a helpful assistant" in formatted
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted
        assert "User: What's the weather?" in formatted
        assert "Assistant:" in formatted
    
    def test_format_completion_prompt(self, prompt_manager):
        """Test formatting completion prompt."""
        examples = [
            {"input": "2 + 2", "output": "4"},
            {"input": "5 * 3", "output": "15"}
        ]
        
        formatted = prompt_manager.format_prompt(
            PromptFormat.COMPLETION,
            "10 / 2",
            examples=examples
        )
        
        assert "Input: 2 + 2" in formatted
        assert "Output: 4" in formatted
        assert "Input: 10 / 2" in formatted
        assert "Output:" in formatted
    
    def test_delete_template(self, prompt_manager):
        """Test deleting templates."""
        # Create template
        prompt_manager.create_template(
            name="to_delete",
            template="Delete me: {{ input }}",
            format=PromptFormat.INSTRUCTION,
            save=False
        )
        
        # Verify it exists
        assert prompt_manager.get_template("to_delete") is not None
        
        # Delete it
        success = prompt_manager.delete_template("to_delete")
        assert success is True
        
        # Verify it's gone
        with pytest.raises(KeyError):
            prompt_manager.get_template("to_delete")
    
    def test_get_template_versions(self, prompt_manager):
        """Test getting template versions."""
        # Create multiple versions
        prompt_manager.create_template(
            name="versioned",
            template="Version 1: {{ input }}",
            format=PromptFormat.INSTRUCTION,
            version="1.0.0",
            save=False
        )
        prompt_manager.create_template(
            name="versioned",
            template="Version 2: {{ input }}",
            format=PromptFormat.INSTRUCTION,
            version="2.0.0",
            save=False
        )
        
        versions = prompt_manager.get_template_versions("versioned")
        assert "1.0.0" in versions
        assert "2.0.0" in versions
        assert len(versions) == 2