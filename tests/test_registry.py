"""Tests for component registry functionality."""

import pytest
from unittest.mock import Mock

from mlpipeline.core.registry import ComponentRegistry, component_registry
from mlpipeline.core.interfaces import PipelineComponent, ComponentType


class MockComponent(PipelineComponent):
    """Mock component for testing."""
    
    def __init__(self, component_name: str = "mock", **kwargs):
        super().__init__(ComponentType.DATA_PREPROCESSING)
        self.component_name = component_name
        self.kwargs = kwargs
    
    def execute(self, context):
        """Mock execute method."""
        return Mock(success=True, artifacts=[], metrics={}, metadata={})
    
    def validate_config(self, config):
        """Mock validate config method."""
        return True


class InvalidComponent:
    """Invalid component that doesn't inherit from PipelineComponent."""
    pass


class TestComponentRegistry:
    """Test component registry functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ComponentRegistry()
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        assert len(self.registry._components) == 0
        assert len(self.registry._instances) == 0
        assert self.registry.logger is not None
    
    def test_register_valid_component(self):
        """Test registering a valid component."""
        self.registry.register_component("test_component", MockComponent)
        
        assert "test_component" in self.registry._components
        assert self.registry._components["test_component"] == MockComponent
    
    def test_register_invalid_component(self):
        """Test registering an invalid component."""
        with pytest.raises(ValueError, match="must inherit from PipelineComponent"):
            self.registry.register_component("invalid_component", InvalidComponent)
    
    def test_get_component_class_exists(self):
        """Test getting an existing component class."""
        self.registry.register_component("test_component", MockComponent)
        
        component_class = self.registry.get_component_class("test_component")
        assert component_class == MockComponent
    
    def test_get_component_class_not_exists(self):
        """Test getting a non-existent component class."""
        component_class = self.registry.get_component_class("nonexistent")
        assert component_class is None
    
    def test_create_component_success(self):
        """Test creating a component instance successfully."""
        self.registry.register_component("test_component", MockComponent)
        
        instance = self.registry.create_component("test_component", component_name="test_instance")
        
        assert instance is not None
        assert isinstance(instance, MockComponent)
        assert instance.component_name == "test_instance"
        assert "test_component" in self.registry._instances
        assert self.registry._instances["test_component"] == instance
    
    def test_create_component_not_registered(self):
        """Test creating a component that's not registered."""
        instance = self.registry.create_component("nonexistent")
        assert instance is None
    
    def test_create_component_with_exception(self):
        """Test creating a component that raises an exception."""
        class FailingComponent(PipelineComponent):
            def __init__(self, **kwargs):
                if kwargs.get("fail", False):
                    raise ValueError("Component initialization failed")
                super().__init__(ComponentType.DATA_PREPROCESSING)
            
            def execute(self, context):
                return Mock()
            
            def validate_config(self, config):
                return True
        
        self.registry.register_component("failing_component", FailingComponent)
        
        instance = self.registry.create_component("failing_component", fail=True)
        assert instance is None
    
    def test_get_component_instance_exists(self):
        """Test getting an existing component instance."""
        self.registry.register_component("test_component", MockComponent)
        created_instance = self.registry.create_component("test_component")
        
        retrieved_instance = self.registry.get_component_instance("test_component")
        assert retrieved_instance == created_instance
    
    def test_get_component_instance_not_exists(self):
        """Test getting a non-existent component instance."""
        instance = self.registry.get_component_instance("nonexistent")
        assert instance is None
    
    def test_list_components(self):
        """Test listing all registered components."""
        self.registry.register_component("component1", MockComponent)
        self.registry.register_component("component2", MockComponent)
        
        components = self.registry.list_components()
        
        assert len(components) == 2
        assert "component1" in components
        assert "component2" in components
        assert components["component1"] == MockComponent
        assert components["component2"] == MockComponent
        
        # Ensure it returns a copy
        components["component3"] = MockComponent
        assert "component3" not in self.registry._components
    
    def test_unregister_component_exists(self):
        """Test unregistering an existing component."""
        self.registry.register_component("test_component", MockComponent)
        self.registry.create_component("test_component")
        
        # Verify component and instance exist
        assert "test_component" in self.registry._components
        assert "test_component" in self.registry._instances
        
        # Unregister
        result = self.registry.unregister_component("test_component")
        
        assert result is True
        assert "test_component" not in self.registry._components
        assert "test_component" not in self.registry._instances
    
    def test_unregister_component_not_exists(self):
        """Test unregistering a non-existent component."""
        result = self.registry.unregister_component("nonexistent")
        assert result is False
    
    def test_unregister_component_class_only(self):
        """Test unregistering a component that has no instance."""
        self.registry.register_component("test_component", MockComponent)
        
        # Don't create instance
        assert "test_component" in self.registry._components
        assert "test_component" not in self.registry._instances
        
        result = self.registry.unregister_component("test_component")
        
        assert result is True
        assert "test_component" not in self.registry._components
    
    def test_multiple_component_types(self):
        """Test registering components of different types."""
        class DataComponent(PipelineComponent):
            def __init__(self):
                super().__init__(ComponentType.DATA_PREPROCESSING)
            
            def execute(self, context):
                return Mock()
            
            def validate_config(self, config):
                return True
        
        class ModelComponent(PipelineComponent):
            def __init__(self):
                super().__init__(ComponentType.MODEL_TRAINING)
            
            def execute(self, context):
                return Mock()
            
            def validate_config(self, config):
                return True
        
        self.registry.register_component("data_component", DataComponent)
        self.registry.register_component("model_component", ModelComponent)
        
        data_instance = self.registry.create_component("data_component")
        model_instance = self.registry.create_component("model_component")
        
        assert data_instance.component_type == ComponentType.DATA_PREPROCESSING
        assert model_instance.component_type == ComponentType.MODEL_TRAINING
    
    def test_component_with_complex_kwargs(self):
        """Test creating component with complex keyword arguments."""
        class ComplexComponent(PipelineComponent):
            def __init__(self, config_dict=None, param_list=None, **kwargs):
                super().__init__(ComponentType.DATA_PREPROCESSING)
                self.config_dict = config_dict or {}
                self.param_list = param_list or []
                self.extra_kwargs = kwargs
            
            def execute(self, context):
                return Mock()
            
            def validate_config(self, config):
                return True
        
        self.registry.register_component("complex_component", ComplexComponent)
        
        instance = self.registry.create_component(
            "complex_component",
            config_dict={"key": "value"},
            param_list=[1, 2, 3],
            extra_param="extra_value"
        )
        
        assert instance.config_dict == {"key": "value"}
        assert instance.param_list == [1, 2, 3]
        assert instance.extra_kwargs["extra_param"] == "extra_value"
    
    def test_registry_thread_safety(self):
        """Test registry operations in concurrent scenarios."""
        import threading
        import time
        
        results = []
        errors = []
        
        def register_and_create(component_name):
            try:
                self.registry.register_component(component_name, MockComponent)
                time.sleep(0.01)  # Small delay to increase chance of race condition
                instance = self.registry.create_component(component_name, component_name=component_name)
                results.append((component_name, instance is not None))
            except Exception as e:
                errors.append((component_name, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_and_create, args=(f"component_{i}",))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(success for _, success in results)
        
        # Verify all components were registered
        components = self.registry.list_components()
        assert len(components) == 5
        for i in range(5):
            assert f"component_{i}" in components


class TestGlobalComponentRegistry:
    """Test the global component registry instance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear global registry for clean tests
        component_registry._components.clear()
        component_registry._instances.clear()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clear global registry after tests
        component_registry._components.clear()
        component_registry._instances.clear()
    
    def test_global_registry_singleton(self):
        """Test that global registry is a singleton."""
        from mlpipeline.core.registry import component_registry as registry1
        from mlpipeline.core.registry import component_registry as registry2
        
        assert registry1 is registry2
        assert registry1 is component_registry
    
    def test_global_registry_persistence(self):
        """Test that global registry persists across imports."""
        # Register component
        component_registry.register_component("persistent_component", MockComponent)
        
        # Import registry again
        from mlpipeline.core.registry import component_registry as new_registry
        
        # Should still have the component
        assert "persistent_component" in new_registry._components
        assert new_registry.get_component_class("persistent_component") == MockComponent
    
    def test_global_registry_operations(self):
        """Test basic operations on global registry."""
        # Register component
        component_registry.register_component("global_test", MockComponent)
        
        # Create instance
        instance = component_registry.create_component("global_test", component_name="global_instance")
        
        assert instance is not None
        assert instance.component_name == "global_instance"
        
        # List components
        components = component_registry.list_components()
        assert "global_test" in components
        
        # Unregister
        result = component_registry.unregister_component("global_test")
        assert result is True
        
        # Verify removal
        assert component_registry.get_component_class("global_test") is None


class TestComponentRegistryEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ComponentRegistry()
    
    def test_register_none_component(self):
        """Test registering None as component."""
        with pytest.raises(TypeError):
            self.registry.register_component("none_component", None)
    
    def test_register_empty_name(self):
        """Test registering component with empty name."""
        self.registry.register_component("", MockComponent)
        
        # Should work, but not recommended
        assert "" in self.registry._components
    
    def test_register_duplicate_component(self):
        """Test registering component with duplicate name."""
        self.registry.register_component("duplicate", MockComponent)
        
        # Register again with same name - should overwrite
        class AnotherComponent(PipelineComponent):
            def __init__(self):
                super().__init__(ComponentType.MODEL_TRAINING)
            
            def execute(self, context):
                return Mock()
            
            def validate_config(self, config):
                return True
        
        self.registry.register_component("duplicate", AnotherComponent)
        
        # Should have the new component
        component_class = self.registry.get_component_class("duplicate")
        assert component_class == AnotherComponent
    
    def test_create_component_with_none_name(self):
        """Test creating component with None name."""
        self.registry.register_component("test", MockComponent)
        
        instance = self.registry.create_component(None)
        assert instance is None
    
    def test_large_number_of_components(self):
        """Test registry with large number of components."""
        num_components = 1000
        
        # Register many components
        for i in range(num_components):
            self.registry.register_component(f"component_{i}", MockComponent)
        
        # Verify all were registered
        components = self.registry.list_components()
        assert len(components) == num_components
        
        # Test retrieval
        for i in range(0, num_components, 100):  # Test every 100th component
            component_class = self.registry.get_component_class(f"component_{i}")
            assert component_class == MockComponent
        
        # Test creation of some instances
        for i in range(0, 10):
            instance = self.registry.create_component(f"component_{i}")
            assert instance is not None
    
    def test_component_with_abstract_methods(self):
        """Test registering component with abstract methods."""
        from abc import ABC, abstractmethod
        
        class AbstractComponent(PipelineComponent, ABC):
            def __init__(self):
                super().__init__(ComponentType.DATA_PREPROCESSING)
            
            @abstractmethod
            def custom_method(self):
                pass
            
            def execute(self, context):
                return Mock()
            
            def validate_config(self, config):
                return True
        
        # Should be able to register abstract component
        self.registry.register_component("abstract", AbstractComponent)
        
        # But creation should fail due to abstract methods
        instance = self.registry.create_component("abstract")
        assert instance is None  # Should fail to create due to abstract methods
    
    def test_component_inheritance_hierarchy(self):
        """Test components with inheritance hierarchy."""
        class BaseComponent(PipelineComponent):
            def __init__(self, base_param="base"):
                super().__init__(ComponentType.DATA_PREPROCESSING)
                self.base_param = base_param
            
            def execute(self, context):
                return Mock()
            
            def validate_config(self, config):
                return True
        
        class DerivedComponent(BaseComponent):
            def __init__(self, derived_param="derived", **kwargs):
                super().__init__(**kwargs)
                self.derived_param = derived_param
        
        self.registry.register_component("base", BaseComponent)
        self.registry.register_component("derived", DerivedComponent)
        
        base_instance = self.registry.create_component("base", base_param="custom_base")
        derived_instance = self.registry.create_component(
            "derived", 
            base_param="custom_base", 
            derived_param="custom_derived"
        )
        
        assert base_instance.base_param == "custom_base"
        assert derived_instance.base_param == "custom_base"
        assert derived_instance.derived_param == "custom_derived"
        assert isinstance(derived_instance, BaseComponent)
        assert isinstance(derived_instance, DerivedComponent)