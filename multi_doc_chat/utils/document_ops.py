from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import subprocess
import shutil
from langchain.schema import Document
from multi_doc_chat.logger.custom_logger import CustomLogger as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from fastapi import UploadFile

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using appropriate loader based on extension."""
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
                try:
                    # Primary approach using PyPDFLoader (pypdf)
                    pdf_docs = loader.load()
                    docs.extend(pdf_docs)
                    continue
                except Exception as pdf_exc:
                    # pypdf can fail on malformed / unusual cmap tables (UnboundLocalError)
                    log.warning("Primary PDF loader failed, attempting pdftotext fallback", path=str(p), error=str(pdf_exc))
                    # Fall back to system pdftotext (requires poppler-utils)
                    if shutil.which("pdftotext"):
                        try:
                            text = _extract_text_with_pdftotext(p)
                            # Create a single Document with full text
                            docs.append(Document(page_content=text, metadata={"source": str(p)}))
                            continue
                        except Exception as fallback_exc:
                            log.error("pdftotext fallback also failed", path=str(p), error=str(fallback_exc))
                            raise
                    else:
                        log.error("pdftotext not available on system, cannot fallback", path=str(p))
                        raise
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e
    

class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


def _extract_text_with_pdftotext(path: Path) -> str:
    """Use system pdftotext to extract text from a PDF file. Returns full text."""
    # Create a temporary output file path
    out = Path(str(path) + ".txt")
    try:
        # -layout preserves original layout better; suppress messages
        subprocess.run(["pdftotext", "-layout", str(path), str(out)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text = out.read_text(encoding="utf-8", errors="ignore")
        return text
    finally:
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass