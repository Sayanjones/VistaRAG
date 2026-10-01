# Chroma needs a newer SQLite than some Linux distros ship.
# If pysqlite3 is installed, use it; otherwise carry on with the system one.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from typing import Any, Dict

from dotenv import load_dotenv
from PIL import Image
from img2vec_pytorch import Img2Vec
from chromadb import EmbeddingFunction, Embeddings
from chromadb.api.types import Images
from chromadb.utils.embedding_functions import register_embedding_function

load_dotenv()

DB_PATH = "./db"
DATA_PATH = "./data"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"


@register_embedding_function
class ImageEmbeddings(EmbeddingFunction):
    """Turns images (RGB numpy arrays) into vectors using a pretrained ResNet."""

    def __init__(self):
        self.model = Img2Vec()

    def __call__(self, input: Images) -> Embeddings:
        return [self.model.get_vec(Image.fromarray(img)) for img in input]

    @staticmethod
    def name() -> str:
        return "img2vec"

    def get_config(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return ImageEmbeddings()
