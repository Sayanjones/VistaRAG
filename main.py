from utils import DB_PATH  # keep first: it applies the SQLite fix before chromadb loads

import chromadb
import streamlit as st

from query_data import classify_img, get_most_similar_chunks, create_response

st.set_page_config(page_title="Tour Guide AI Assistant")

client_db = chromadb.PersistentClient(path=DB_PATH)

st.title("Tour Guide AI Assistant")
st.header("Upload a photo")

file = st.file_uploader("Photo", type=["jpeg", "jpg", "png"], label_visibility="collapsed")

if file:
    st.image(file, use_container_width=True)

    category = classify_img(client_db, file)
    name = category.replace("_", " ").title()
    st.write(f"This looks like **{name}**. Want to know anything about it?")

    question = st.text_input(f"Ask a question about {name}:")
    if question:
        with st.spinner("Thinking..."):
            chunks, metadata = get_most_similar_chunks(client_db, question, category)
            answer, sources = create_response(chunks, metadata, question)
        st.write(answer)
        with st.expander("Sources"):
            for m in sources:
                st.write(m.get("source", "unknown"))
