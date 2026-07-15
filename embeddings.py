"""Embedding adapter shared by ingestion and retrieval."""

from pathlib import Path

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_core.embeddings import Embeddings


class ChromaMiniLMEmbeddings(Embeddings):
    """Use Chroma's ONNX all-MiniLM-L6-v2 model without PyTorch."""

    def __init__(self) -> None:
        self._embedding_function = ONNXMiniLM_L6_V2()
        self._embedding_function.DOWNLOAD_PATH = (
            Path(__file__).resolve().parent
            / ".chroma_cache"
            / "onnx_models"
            / self._embedding_function.MODEL_NAME
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._embedding_function(texts)
        return [[float(value) for value in embedding] for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
