"""Prompt management system for few-shot learning."""

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jinja2 import Environment, FileSystemLoader, Template, TemplateError
from pydantic import BaseModel, Field, field_validator


class PromptFormat(str, Enum):
    """Supported prompt formats."""
    INSTRUCTION = "instruction"
    CHAT = "chat"
    COMPLETION = "completion"


class PromptTemplate(BaseModel):
    """Prompt template model."""
    name: str
    version: str = "1.0.0"
    format: PromptFormat
    template: str
    variables: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    
    @field_validator('template')
    @classmethod
    def validate_template(cls, v):
        """Validate template syntax."""
        try:
            Template(v)
        except TemplateError as e:
            raise ValueError(f"Invalid template syntax: {e}")
        return v
    
    def __init__(self, **data):
        """Initialize and extract variables if not provided."""
        if 'variables' not in data or not data['variables']:
            if 'template' in data:
                template = data['template']
                # Extract Jinja2 variables
                variables = re.findall(r'\{\{\s*(\w+)\s*\}\}', template)
                data['variables'] = list(set(variables))
        super().__init__(**data)


class PromptManager:
    """Manages prompt templates with versioning and validation."""
    
    def __init__(self, templates_dir: Union[str, Path] = "templates"):
        """Initialize prompt manager.
        
        Args:
            templates_dir: Directory containing prompt templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates from the templates directory."""
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                
                template = PromptTemplate(**template_data)
                if template.name not in self._templates:
                    self._templates[template.name] = {}
                self._templates[template.name][template.version] = template
                
            except Exception as e:
                print(f"Warning: Failed to load template {template_file}: {e}")
    
    def create_template(
        self,
        name: str,
        template: str,
        format: PromptFormat,
        version: str = "1.0.0",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        save: bool = True
    ) -> PromptTemplate:
        """Create a new prompt template.
        
        Args:
            name: Template name
            template: Template string with Jinja2 syntax
            format: Prompt format type
            version: Template version
            description: Optional description
            tags: Optional tags for categorization
            save: Whether to save template to disk
            
        Returns:
            Created PromptTemplate instance
        """
        prompt_template = PromptTemplate(
            name=name,
            version=version,
            format=format,
            template=template,
            description=description,
            tags=tags or []
        )
        
        # Store in memory
        if name not in self._templates:
            self._templates[name] = {}
        self._templates[name][version] = prompt_template
        
        # Save to disk if requested
        if save:
            self._save_template(prompt_template)
        
        return prompt_template
    
    def _save_template(self, template: PromptTemplate) -> None:
        """Save template to disk."""
        filename = f"{template.name}_{template.version}.yaml"
        filepath = self.templates_dir / filename
        
        template_dict = template.model_dump()
        template_dict['created_at'] = template_dict['created_at'].isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(template_dict, f, default_flow_style=False, sort_keys=False)
    
    def get_template(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """Get a template by name and version.
        
        Args:
            name: Template name
            version: Template version (latest if None)
            
        Returns:
            PromptTemplate instance
            
        Raises:
            KeyError: If template not found
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        
        versions = self._templates[name]
        if not versions:
            raise KeyError(f"No versions found for template '{name}'")
        
        if version is None:
            # Get latest version
            version = max(versions.keys())
        
        if version not in versions:
            available_versions = list(versions.keys())
            raise KeyError(f"Version '{version}' not found for template '{name}'. Available: {available_versions}")
        
        return versions[version]
    
    def list_templates(self, format_filter: Optional[PromptFormat] = None) -> List[Dict[str, Any]]:
        """List all available templates.
        
        Args:
            format_filter: Optional format filter
            
        Returns:
            List of template metadata
        """
        templates = []
        for name, versions in self._templates.items():
            for version, template in versions.items():
                if format_filter is None or template.format == format_filter:
                    templates.append({
                        'name': name,
                        'version': version,
                        'format': template.format,
                        'description': template.description,
                        'variables': template.variables,
                        'tags': template.tags,
                        'created_at': template.created_at
                    })
        
        return sorted(templates, key=lambda x: (x['name'], x['version']))
    
    def render_template(
        self,
        name: str,
        variables: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """Render a template with provided variables.
        
        Args:
            name: Template name
            variables: Variables to substitute
            version: Template version (latest if None)
            
        Returns:
            Rendered template string
            
        Raises:
            KeyError: If template not found
            TemplateError: If rendering fails
        """
        template = self.get_template(name, version)
        
        # Validate required variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        try:
            jinja_template = Template(template.template)
            return jinja_template.render(**variables)
        except TemplateError as e:
            raise TemplateError(f"Failed to render template '{name}': {e}")
    
    def validate_template(self, template_str: str) -> Dict[str, Any]:
        """Validate template syntax and extract metadata.
        
        Args:
            template_str: Template string to validate
            
        Returns:
            Validation result with extracted variables
        """
        try:
            jinja_template = Template(template_str)
            variables = re.findall(r'\{\{\s*(\w+)\s*\}\}', template_str)
            
            return {
                'valid': True,
                'variables': list(set(variables)),
                'error': None
            }
        except TemplateError as e:
            return {
                'valid': False,
                'variables': [],
                'error': str(e)
            }
    
    def delete_template(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a template.
        
        Args:
            name: Template name
            version: Template version (all versions if None)
            
        Returns:
            True if deleted successfully
        """
        if name not in self._templates:
            return False
        
        if version is None:
            # Delete all versions
            for v in list(self._templates[name].keys()):
                self._delete_template_file(name, v)
            del self._templates[name]
        else:
            if version not in self._templates[name]:
                return False
            self._delete_template_file(name, version)
            del self._templates[name][version]
            
            # Remove name entry if no versions left
            if not self._templates[name]:
                del self._templates[name]
        
        return True
    
    def _delete_template_file(self, name: str, version: str) -> None:
        """Delete template file from disk."""
        filename = f"{name}_{version}.yaml"
        filepath = self.templates_dir / filename
        if filepath.exists():
            filepath.unlink()
    
    def get_template_versions(self, name: str) -> List[str]:
        """Get all versions of a template.
        
        Args:
            name: Template name
            
        Returns:
            List of version strings
        """
        if name not in self._templates:
            return []
        return list(self._templates[name].keys())
    
    def format_prompt(
        self,
        format_type: PromptFormat,
        content: str,
        system_message: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format content according to prompt format.
        
        Args:
            format_type: Type of prompt format
            content: Main content
            system_message: Optional system message
            examples: Optional few-shot examples
            
        Returns:
            Formatted prompt string
        """
        if format_type == PromptFormat.INSTRUCTION:
            return self._format_instruction(content, system_message, examples)
        elif format_type == PromptFormat.CHAT:
            return self._format_chat(content, system_message, examples)
        elif format_type == PromptFormat.COMPLETION:
            return self._format_completion(content, examples)
        else:
            raise ValueError(f"Unsupported prompt format: {format_type}")
    
    def _format_instruction(
        self,
        content: str,
        system_message: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format as instruction prompt."""
        parts = []
        
        if system_message:
            parts.append(f"System: {system_message}")
        
        if examples:
            parts.append("Examples:")
            for i, example in enumerate(examples, 1):
                parts.append(f"Example {i}:")
                for key, value in example.items():
                    parts.append(f"{key}: {value}")
                parts.append("")
        
        parts.append(f"Instruction: {content}")
        
        return "\n".join(parts)
    
    def _format_chat(
        self,
        content: str,
        system_message: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format as chat prompt."""
        messages = []
        
        if system_message:
            messages.append(f"System: {system_message}")
        
        if examples:
            for example in examples:
                if 'user' in example and 'assistant' in example:
                    messages.append(f"User: {example['user']}")
                    messages.append(f"Assistant: {example['assistant']}")
        
        messages.append(f"User: {content}")
        messages.append("Assistant:")
        
        return "\n".join(messages)
    
    def _format_completion(
        self,
        content: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format as completion prompt."""
        parts = []
        
        if examples:
            for example in examples:
                if 'input' in example and 'output' in example:
                    parts.append(f"Input: {example['input']}")
                    parts.append(f"Output: {example['output']}")
                    parts.append("")
        
        parts.append(f"Input: {content}")
        parts.append("Output:")
        
        return "\n".join(parts)