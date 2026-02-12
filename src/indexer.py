import os
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Document, 
    load_index_from_storage,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.node_parser import TokenTextSplitter
from multiprocessing import cpu_count

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever):
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        # Simple ensemble: combine and deduplicate
        all_nodes = {node.node.node_id: node for node in vector_nodes}
        for node in bm25_nodes:
            if node.node.node_id not in all_nodes:
                all_nodes[node.node.node_id] = node
            else:
                # Average the scores (naive)
                all_nodes[node.node.node_id].score = (all_nodes[node.node.node_id].score + node.score) / 2
        
        return sorted(all_nodes.values(), key=lambda x: x.score, reverse=True)

class RepoIndexer:
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            storage_dir = project_root / "data" / "index"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index: Optional[VectorStoreIndex] = None
        self.vector_store: Optional[FaissVectorStore] = None

    def _initialize_empty_index(self):
        d = 1536  # Default dimension for OpenAI embeddings
        faiss_index = faiss.IndexFlatL2(d)
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))

    def load_or_create(self):
        if (self.storage_dir / "vector_store.json").exists():
            vector_store = FaissVectorStore.from_persist_dir(str(self.storage_dir))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=str(self.storage_dir)
            )
            self.index = load_index_from_storage(storage_context)
            self.vector_store = vector_store
        else:
            self._initialize_empty_index()

    def update_index(self, manifest: Manifest, repo_path: Path):
        self.load_or_create()
        
        # Further Optimize: Use SentenceSplitter but disable slow features, 
        # or stick to a fast tokenizer-based approach.
        # Actually, let's use a very basic character-based splitter if token-based is still slow.
        # But TokenTextSplitter should be fast. Let's ensure it's used correctly.
        from llama_index.core.node_parser import TokenTextSplitter
        parser = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
        
        # Track indexed documents via metadata hashes
        existing_docs = self.index.docstore.docs
        indexed_hashes = {doc.metadata.get("path"): doc.metadata.get("hash") 
                         for doc in existing_docs.values() if doc.metadata.get("path")}
        
        documents = []
        new_or_modified = 0
        
        for rel_path, entry in manifest.files.items():
            # Check if file has changed
            if entry.language and indexed_hashes.get(rel_path) != entry.hash:
                abs_path = repo_path / rel_path
                try:
                    with open(abs_path, "r", errors="ignore") as f:
                        text = f.read()
                    doc = Document(
                        text=text,
                        metadata={
                            "path": rel_path,
                            "language": entry.language,
                            "hash": entry.hash
                        },
                        id_=rel_path
                    )
                    documents.append(doc)
                    new_or_modified += 1
                except Exception as e:
                    print(f"Error reading {abs_path}: {e}")

        if documents:
            print(f"Updating index with {new_or_modified} new/modified documents...")
            # Disable progress bar and overhead during bulk transformation
            # Setting globally might be more reliable
            Settings.node_parser = parser
            
            self.index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=self.index.storage_context,
                show_progress=False, # Disable for speed
                num_workers=cpu_count()
            )
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
        else:
            print("No changes detected. Index is up to date.")

    def get_retriever(self, similarity_top_k: int = 5) -> HybridRetriever:
        if not self.index:
            raise ValueError("Index not loaded")
        
        nodes = list(self.index.docstore.docs.values())
        vector_retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        
        if not nodes:
            # Fallback for empty index
            return HybridRetriever(vector_retriever, vector_retriever) # Simple fallback
            
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=similarity_top_k
        )
        
        return HybridRetriever(vector_retriever, bm25_retriever)
