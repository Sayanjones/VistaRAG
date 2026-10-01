"""Build the vector databases from everything inside ./data.

Run this once, and again whenever you add or change files in ./data.
"""
import glob
import os

from utils import ImageEmbeddings, DATA_PATH, DB_PATH, TEXT_EMBEDDING_MODEL

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    client = chromadb.PersistentClient(path=DB_PATH)

    img_collection = client.get_or_create_collection(
        name="imgs",
        embedding_function=ImageEmbeddings(),
        data_loader=ImageLoader(),
    )
    text_ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=TEXT_EMBEDDING_MODEL,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, add_start_index=True
    )

    categories = [
        d for d in sorted(os.listdir(DATA_PATH))
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ]
    if not categories:
        raise SystemExit("No folders found in ./data. See the README for the expected layout.")

    for category in categories:
        folder = os.path.join(DATA_PATH, category)
        print(f"-> {category}")

        # Images: the category name is baked into the ID so we can read it back later.
        img_paths = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in glob.glob(os.path.join(folder, ext))
        )
        if img_paths:
            img_collection.upsert(
                ids=[f"{category}-{os.path.basename(p)}" for p in img_paths],
                uris=img_paths,
            )
        print(f"   {len(img_paths)} images")

        # Text: split into overlapping chunks and embed.
        docs = client.get_or_create_collection(
            name=f"documents_{category}", embedding_function=text_ef
        )
        ids, texts, metas = [], [], []
        for txt_path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
            with open(txt_path, encoding="utf-8") as f:
                content = f.read()
            pieces = splitter.create_documents([content], metadatas=[{"source": txt_path}])
            for i, piece in enumerate(pieces):
                ids.append(f"{os.path.basename(txt_path)}-{i}")
                texts.append(piece.page_content)
                metas.append(piece.metadata)
        if texts:
            docs.upsert(ids=ids, documents=texts, metadatas=metas)
        print(f"   {len(texts)} text chunks")

    print("Done.")


if __name__ == "__main__":
    main()
