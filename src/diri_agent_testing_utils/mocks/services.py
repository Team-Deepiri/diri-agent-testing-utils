"""
In-memory mock implementations for common external services.

These mocks require no external dependencies and provide basic
implementations suitable for unit testing agent systems.
"""
import time
from typing import Any, Dict, List, Optional


class MockRedis:
    """In-memory Redis mock with basic get/set/delete/exists/expire support."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

    def _check_expiry(self, key: str) -> bool:
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
            return True
        return False

    def get(self, key: str) -> Optional[str]:
        self._check_expiry(key)
        return self._store.get(key)

    def set(
        self, key: str, value: Any, ex: Optional[int] = None, **kwargs: Any
    ) -> bool:
        self._store[key] = value
        if ex is not None:
            self._expiry[key] = time.time() + ex
        return True

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                self._expiry.pop(key, None)
                count += 1
        return count

    def exists(self, *keys: str) -> int:
        count = 0
        for key in keys:
            self._check_expiry(key)
            if key in self._store:
                count += 1
        return count

    def expire(self, key: str, seconds: int) -> bool:
        if key in self._store:
            self._expiry[key] = time.time() + seconds
            return True
        return False

    def keys(self, pattern: str = "*") -> List[str]:
        # Simple pattern support: only "*" (all keys)
        if pattern == "*":
            return list(self._store.keys())
        # Basic prefix matching for "prefix*"
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._store if k.startswith(prefix)]
        return [k for k in self._store if k == pattern]

    def flushdb(self) -> None:
        self._store.clear()
        self._expiry.clear()


class MockVectorStore:
    """In-memory vector store mock for testing.

    Stores documents as dicts and returns them on similarity_search.
    No real embeddings — returns all stored docs (up to k).
    """

    def __init__(self) -> None:
        self._documents: List[Dict[str, Any]] = []

    def add_documents(
        self, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[str]:
        """Add documents to the store. Each doc should have 'content' and optional 'metadata'."""
        ids = []
        for doc in documents:
            doc_id = str(len(self._documents))
            self._documents.append({
                "id": doc_id,
                "content": doc.get("content", doc.get("page_content", "")),
                "metadata": doc.get("metadata", {}),
            })
            ids.append(doc_id)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Return up to k stored documents. Optionally filters by query substring."""
        results = []
        for doc in self._documents:
            if query.lower() in doc["content"].lower() or not query.strip():
                results.append(doc)
            if len(results) >= k:
                break
        # If no substring matches, return first k docs
        if not results:
            results = self._documents[:k]
        return results

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Async variant of similarity_search."""
        return self.similarity_search(query, k, **kwargs)

    def stats(self) -> Dict[str, Any]:
        return {
            "collection_name": "mock",
            "num_entities": len(self._documents),
            "dimension": 384,
        }

    def reset(self) -> None:
        self._documents.clear()


class MockMemoryManager:
    """In-memory mock for agent memory management."""

    def __init__(self) -> None:
        self._memories: List[Dict[str, Any]] = []

    async def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        session_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a memory entry and return its ID."""
        memory_id = str(len(self._memories))
        self._memories.append({
            "id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "session_id": session_id,
            "importance": importance,
            "metadata": metadata or {},
        })
        return memory_id

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories by query substring and optional filters."""
        results = []
        for memory in self._memories:
            if memory_type and memory["memory_type"] != memory_type:
                continue
            if session_id and memory["session_id"] != session_id:
                continue
            if query.lower() in memory["content"].lower():
                results.append(memory)
            if len(results) >= limit:
                break
        return results

    async def build_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_items: int = 10,
    ) -> Dict[str, Any]:
        """Build context from stored memories."""
        memories = await self.search_memories(
            query, session_id=session_id, limit=max_items
        )
        return {
            "memories": memories,
            "count": len(memories),
            "session_id": session_id,
        }

    def reset(self) -> None:
        self._memories.clear()
