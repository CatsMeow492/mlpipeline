"""Example store for few-shot learning."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class Example(BaseModel):
    """Few-shot example model."""
    id: Optional[str] = None
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {"arbitrary_types_allowed": True}


class ExampleStore:
    """Manages few-shot examples with persistence and retrieval."""
    
    def __init__(self, db_path: Union[str, Path] = "examples.db"):
        """Initialize example store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS examples (
                    id TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    metadata TEXT,
                    tags TEXT,
                    embedding BLOB,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON examples(tags)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON examples(created_at)
            """)
    
    def add_example(
        self,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        example_id: Optional[str] = None
    ) -> str:
        """Add a new example to the store.
        
        Args:
            input_text: Input text for the example
            output_text: Expected output text
            metadata: Optional metadata dictionary
            tags: Optional tags for categorization
            embedding: Optional embedding vector
            example_id: Optional custom ID
            
        Returns:
            Example ID
        """
        example = Example(
            id=example_id or self._generate_id(),
            input_text=input_text,
            output_text=output_text,
            metadata=metadata or {},
            tags=tags or [],
            embedding=embedding
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO examples 
                (id, input_text, output_text, metadata, tags, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                example.id,
                example.input_text,
                example.output_text,
                json.dumps(example.metadata),
                json.dumps(example.tags),
                self._serialize_embedding(example.embedding),
                example.created_at.isoformat()
            ))
        
        return example.id
    
    def get_example(self, example_id: str) -> Optional[Example]:
        """Get an example by ID.
        
        Args:
            example_id: Example ID
            
        Returns:
            Example instance or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, input_text, output_text, metadata, tags, embedding, created_at
                FROM examples WHERE id = ?
            """, (example_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_example(row)
    
    def list_examples(
        self,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Example]:
        """List examples with optional filtering.
        
        Args:
            tags: Optional tags to filter by
            limit: Optional limit on number of results
            offset: Offset for pagination
            
        Returns:
            List of examples
        """
        query = "SELECT id, input_text, output_text, metadata, tags, embedding, created_at FROM examples"
        params = []
        
        if tags:
            # Simple tag filtering - could be improved with proper JSON queries
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            query += " WHERE " + " AND ".join(tag_conditions)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        if offset:
            query += " OFFSET ?"
            params.append(offset)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_example(row) for row in cursor.fetchall()]
    
    def delete_example(self, example_id: str) -> bool:
        """Delete an example.
        
        Args:
            example_id: Example ID
            
        Returns:
            True if deleted successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM examples WHERE id = ?", (example_id,))
            return cursor.rowcount > 0
    
    def update_embedding(self, example_id: str, embedding: List[float]) -> bool:
        """Update embedding for an example.
        
        Args:
            example_id: Example ID
            embedding: New embedding vector
            
        Returns:
            True if updated successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE examples SET embedding = ? WHERE id = ?
            """, (self._serialize_embedding(embedding), example_id))
            return cursor.rowcount > 0
    
    def get_examples_with_embeddings(self) -> List[Example]:
        """Get all examples that have embeddings.
        
        Returns:
            List of examples with embeddings
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, input_text, output_text, metadata, tags, embedding, created_at
                FROM examples WHERE embedding IS NOT NULL
            """)
            return [self._row_to_example(row) for row in cursor.fetchall()]
    
    def count_examples(self, tags: Optional[List[str]] = None) -> int:
        """Count examples with optional tag filtering.
        
        Args:
            tags: Optional tags to filter by
            
        Returns:
            Number of examples
        """
        query = "SELECT COUNT(*) FROM examples"
        params = []
        
        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            query += " WHERE " + " AND ".join(tag_conditions)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]
    
    def _generate_id(self) -> str:
        """Generate a unique ID for an example."""
        import uuid
        return str(uuid.uuid4())
    
    def _serialize_embedding(self, embedding: Optional[List[float]]) -> Optional[bytes]:
        """Serialize embedding vector to bytes."""
        if embedding is None:
            return None
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, embedding_bytes: Optional[bytes]) -> Optional[List[float]]:
        """Deserialize embedding vector from bytes."""
        if embedding_bytes is None:
            return None
        return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
    
    def _row_to_example(self, row: Tuple) -> Example:
        """Convert database row to Example instance."""
        return Example(
            id=row[0],
            input_text=row[1],
            output_text=row[2],
            metadata=json.loads(row[3]) if row[3] else {},
            tags=json.loads(row[4]) if row[4] else [],
            embedding=self._deserialize_embedding(row[5]),
            created_at=datetime.fromisoformat(row[6])
        )
    
    def export_examples(self, filepath: Union[str, Path], format: str = "json") -> None:
        """Export examples to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'jsonl')
        """
        examples = self.list_examples()
        filepath = Path(filepath)
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([example.model_dump() for example in examples], f, indent=2, default=str)
        elif format == "jsonl":
            with open(filepath, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example.model_dump(), default=str) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_examples(self, filepath: Union[str, Path], format: str = "json") -> int:
        """Import examples from file.
        
        Args:
            filepath: Input file path
            format: Import format ('json' or 'jsonl')
            
        Returns:
            Number of examples imported
        """
        filepath = Path(filepath)
        count = 0
        
        if format == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.add_example(
                        input_text=item['input_text'],
                        output_text=item['output_text'],
                        metadata=item.get('metadata', {}),
                        tags=item.get('tags', []),
                        embedding=item.get('embedding'),
                        example_id=item.get('id')
                    )
                    count += 1
        elif format == "jsonl":
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.add_example(
                        input_text=item['input_text'],
                        output_text=item['output_text'],
                        metadata=item.get('metadata', {}),
                        tags=item.get('tags', []),
                        embedding=item.get('embedding'),
                        example_id=item.get('id')
                    )
                    count += 1
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return count