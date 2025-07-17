"""Example demonstrating few-shot learning capabilities."""

import tempfile
from pathlib import Path

from mlpipeline.few_shot import (
    PromptManager,
    ExampleStore,
    SimilarityEngine,
    FewShotInferencePipeline,
    PromptFormat,
    OpenAICompatibleClient
)


def main():
    """Demonstrate few-shot learning pipeline."""
    print("Few-Shot Learning Pipeline Example")
    print("=" * 40)
    
    # Create temporary directories for this example
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        templates_dir = tmpdir / "templates"
        db_path = tmpdir / "examples.db"
        
        # 1. Set up Prompt Manager
        print("\n1. Setting up Prompt Manager...")
        prompt_manager = PromptManager(templates_dir=templates_dir)
        
        # Create a sentiment classification template
        template = prompt_manager.create_template(
            name="sentiment_classification",
            template="Classify the sentiment of the following text as Positive, Negative, or Neutral:\n\nText: {{ input }}\nSentiment:",
            format=PromptFormat.COMPLETION,
            description="Sentiment classification template",
            tags=["sentiment", "classification"]
        )
        print(f"Created template: {template.name}")
        print(f"Variables: {template.variables}")
        
        # 2. Set up Example Store
        print("\n2. Setting up Example Store...")
        example_store = ExampleStore(db_path=db_path)
        
        # Add training examples
        examples_data = [
            ("I love this product! It's amazing!", "Positive", ["sentiment", "product"]),
            ("This is the worst experience ever.", "Negative", ["sentiment", "experience"]),
            ("The weather is okay today.", "Neutral", ["sentiment", "weather"]),
            ("Fantastic service and great quality!", "Positive", ["sentiment", "service"]),
            ("I hate waiting in long lines.", "Negative", ["sentiment", "waiting"]),
            ("The movie was neither good nor bad.", "Neutral", ["sentiment", "movie"]),
            ("Excellent customer support!", "Positive", ["sentiment", "support"]),
            ("This software is buggy and slow.", "Negative", ["sentiment", "software"]),
        ]
        
        for input_text, output_text, tags in examples_data:
            example_id = example_store.add_example(
                input_text=input_text,
                output_text=output_text,
                tags=tags
            )
            print(f"Added example: {example_id[:8]}...")
        
        print(f"Total examples: {example_store.count_examples()}")
        
        # 3. Set up Similarity Engine
        print("\n3. Setting up Similarity Engine...")
        similarity_engine = SimilarityEngine(
            example_store=example_store,
            use_tfidf_fallback=True
        )
        
        # Get statistics
        stats = similarity_engine.get_example_statistics()
        print(f"Example statistics:")
        print(f"  - Total examples: {stats['total_examples']}")
        print(f"  - Unique tags: {stats['unique_tags']}")
        print(f"  - Tag distribution: {stats['tag_distribution']}")
        
        # 4. Test similarity search
        print("\n4. Testing Similarity Search...")
        query = "This product is incredible!"
        similar_examples = similarity_engine.find_similar_examples(
            query_text=query,
            k=3,
            similarity_threshold=0.1
        )
        
        print(f"Query: '{query}'")
        print("Similar examples:")
        for i, (example, similarity) in enumerate(similar_examples, 1):
            print(f"  {i}. [{similarity:.3f}] {example.input_text} -> {example.output_text}")
        
        # 5. Demonstrate prompt formatting
        print("\n5. Testing Prompt Formatting...")
        
        # Format instruction prompt
        instruction_prompt = prompt_manager.format_prompt(
            PromptFormat.INSTRUCTION,
            "This service is outstanding!",
            system_message="You are a sentiment classifier",
            examples=[
                {"input": "Great product!", "output": "Positive"},
                {"input": "Terrible quality", "output": "Negative"}
            ]
        )
        print("Instruction format:")
        print(instruction_prompt)
        print()
        
        # Format chat prompt
        chat_prompt = prompt_manager.format_prompt(
            PromptFormat.CHAT,
            "This service is outstanding!",
            examples=[
                {"user": "Great product!", "assistant": "Positive"},
                {"user": "Terrible quality", "assistant": "Negative"}
            ]
        )
        print("Chat format:")
        print(chat_prompt)
        print()
        
        # 6. Demonstrate template rendering
        print("6. Testing Template Rendering...")
        rendered = prompt_manager.render_template(
            "sentiment_classification",
            {"input": "This is an awesome day!"}
        )
        print("Rendered template:")
        print(rendered)
        
        # 7. Demonstrate OpenAI-compatible client (mock)
        print("\n7. OpenAI-Compatible Client Example...")
        print("Note: This would require an actual API endpoint")
        
        # Example of how to use the client
        # client = OpenAICompatibleClient(
        #     base_url="http://localhost:8000",
        #     api_key="your-api-key"
        # )
        # response = client.complete("Hello, world!", max_tokens=50)
        
        # 8. Demonstrate few-shot inference pipeline (with mock)
        print("\n8. Few-Shot Inference Pipeline Example...")
        print("Note: This would require transformers library or API endpoint")
        
        try:
            # This would work with an actual OpenAI-compatible API
            pipeline = FewShotInferencePipeline(
                model_name_or_path="http://localhost:8000",  # Mock API
                prompt_manager=prompt_manager,
                example_store=example_store,
                similarity_engine=similarity_engine
            )
            
            print("Pipeline created successfully (OpenAI-compatible mode)")
            print(f"Model info: {pipeline.get_model_info()}")
            
        except Exception as e:
            print(f"Pipeline creation failed (expected): {e}")
        
        # 9. Example augmentation
        print("\n9. Example Augmentation...")
        original_examples = example_store.list_examples(limit=2)
        if original_examples:
            augmented = similarity_engine.augment_examples(
                examples=original_examples,
                augmentation_methods=['paraphrase']
            )
            
            print("Original examples:")
            for ex in original_examples:
                print(f"  - {ex.input_text} -> {ex.output_text}")
            
            print("Augmented examples:")
            for ex in augmented:
                print(f"  - {ex.input_text} -> {ex.output_text}")
        
        # 10. Export/Import examples
        print("\n10. Export/Import Examples...")
        export_file = tmpdir / "examples_export.json"
        example_store.export_examples(export_file, format="json")
        print(f"Exported examples to: {export_file}")
        
        # Create new store and import
        import_db = tmpdir / "import.db"
        import_store = ExampleStore(db_path=import_db)
        count = import_store.import_examples(export_file, format="json")
        print(f"Imported {count} examples to new store")
        
        print("\n" + "=" * 40)
        print("Few-Shot Learning Pipeline Demo Complete!")


if __name__ == "__main__":
    main()