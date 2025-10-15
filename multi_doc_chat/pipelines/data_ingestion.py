from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.logger.custom_logger import CustomLogger as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
import json
import uuid
from datetime import datetime
import hashlib
import sys

def generate_id() -> str:
    """Generate a unique identifier."""
    dt = datetime.now()
    date_part = dt.strftime("%Y%m%d")
    time_part = dt.strftime("%H%M%S%f")
    unique_id = uuid.uuid4().hex[:8]
    # Format: session_YYYYMMDD_HHMMSSffffff_xxxxxxxx -> 4 parts when split on '_'
    return f"session_{date_part}_{time_part}_{unique_id}"


def generate_session_id() -> str:
    """Backward-compatible alias: previously exported as generate_session_id."""
    return generate_id()

class ChatIngestor:
    def __init__(self,
      temp_base: str ="data",
      faiss_base: str ="faiss_index",
      use_session_dirs: bool =True,
      session_id: Optional[str] =None           
    ):
        try:
            self.model_loader = ModelLoader()
            self.use_session_dirs = use_session_dirs
            self.session_id = session_id or generate_id()
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)
            
            log.info("ChatIngestor initialized with session_id: %s", self.session_id)
        except Exception as e:
            log.error("Error initializing ChatIngestor: %s", str(e))
            raise DocumentPortalException(f"Initialization error: {str(e)}") from e
        
    def _resolve_dir(self, base_path: Path) -> Path:
        """Resolve directory path based on session settings."""
        if self.use_session_dirs:
            dir_path = base_path / self.session_id
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path
        return base_path
    
    def _split_documents(self, documents: List[Document], chunk_size: int =1000, chunk_overlap: int =200) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(documents)
            log.info("Documents split into %d chunks", len(split_docs))
            return split_docs
        except Exception as e:
            log.error("Error splitting documents: %s", str(e))
            raise DocumentPortalException(f"Document splitting error: {str(e)}") from e

    # Backwards-compatible alias used by older tests / code
    def _split(self, documents: List[Document], chunk_size: int =1000, chunk_overlap: int =200) -> List[Document]:
        """Alias for _split_documents retained for backward compatibility."""
        return self._split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def built_retriver(self, uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5):
        try:
            path = save_uploaded_files(uploaded_files, self.temp_dir)
            documents = load_documents(path)
            if not documents:
                raise ValueError("No documents were loaded.")

            chunks = self._split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            fm = FaissManager(self.faiss_dir, self.model_loader)
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metadatas)
            except Exception as e:
                vs = fm.load_or_create(texts=texts, metadatas=metadatas)
                
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            # Configure search parameters based on search type
            search_kwargs = {"k": k}
            
            if search_type == "mmr":
                # MMR needs fetch_k (docs to fetch) and lambda_mult (diversity parameter)
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR search", k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
            
            return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e



SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# FAISS Manager (load-or-create)
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}} ## this is dict of rows

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists


        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self)-> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")


    def add_documents(self,docs: List[Document]):

        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")

        new_docs: List[Document] = []

        for d in docs:

            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs


        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs