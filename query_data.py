"""Retrieval and answering. Can also be run directly for a quick test:

    python query_data.py path/to/photo.jpg "What is this building made of?"
"""
import sys

from utils import ImageEmbeddings, DB_PATH, LLM_MODEL

import chromadb
import numpy as np
from openai import OpenAI
from PIL import Image

PROMPT_TEMPLATE = """Answer the question based only on the following context:

{context}

---

Question: {question}
"""


def classify_img(client_db, image_file) -> str:
    """Return the category of the most similar image in the database.

    Note: this always returns the closest match, even for photos of something
    completely unrelated.
    """
    img = np.array(Image.open(image_file).convert("RGB"))
    embedding = ImageEmbeddings()([img])
    collection = client_db.get_collection(name="imgs")
    result = collection.query(query_embeddings=embedding, n_results=1)
    return result["ids"][0][0].split("-")[0]


def get_most_similar_chunks(client_db, question: str, category: str, k: int = 3):
    collection = client_db.get_collection(name=f"documents_{category}")
    result = collection.query(query_texts=[question], n_results=k)
    return result["documents"][0], result["metadatas"][0]


def create_response(chunks, metadata, question: str):
    prompt = PROMPT_TEMPLATE.format(context="\n---\n".join(chunks), question=question)
    response = OpenAI().responses.create(model=LLM_MODEL, input=prompt)
    return response.output_text, metadata


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python query_data.py <image> "<question>"')
    img_path, question = sys.argv[1], sys.argv[2]

    client = chromadb.PersistentClient(path=DB_PATH)
    category = classify_img(client, img_path)
    print(f"Looks like: {category}")
    chunks, meta = get_most_similar_chunks(client, question, category)
    answer, sources = create_response(chunks, meta, question)
    print(answer)
    print(sources)
