import os
import re
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================
# SETTINGS
# =========================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# =========================
# LOAD EMBEDDING MODEL
# =========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================
# GET GROQ API KEY
# =========================

def get_api_key():

    # Streamlit Cloud
    try:

        return st.secrets["GROQ_API_KEY"]

    except Exception:

        pass


    # Local computer
    return os.getenv("GROQ_API_KEY")


# =========================
# EXTRACT PDF TEXT
# =========================

def extract_pdf(pdf_bytes):

    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if text:

                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )

    return pages


# =========================
# CREATE CHUNKS
# =========================

def create_chunks(pages):

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk = text[start:end].strip()

            if chunk:

                chunks.append(
                    {
                        "text": chunk,
                        "page": page_number
                    }
                )

            if end >= len(text):

                break

            start = end - CHUNK_OVERLAP

    return chunks


# =========================
# CREATE EMBEDDINGS
# =========================

def create_embeddings(
    chunks,
    model
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype(
        "float32"
    )


# =========================
# CREATE FAISS DATABASE
# =========================

def create_faiss_index(
    embeddings
):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================
# SEARCH DOCUMENT
# =========================

def search_document(
    question,
    index,
    chunks,
    model
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    k = min(
        TOP_K,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:

            continue

        results.append(
            {
                "text": chunks[index_number]["text"],
                "page": chunks[index_number]["page"],
                "score": float(score)
            }
        )

    return results


# =========================
# GENERATE ANSWER
# =========================

def generate_answer(
    question,
    results
):

    api_key = get_api_key()

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Secrets."
        )


    client = Groq(
        api_key=api_key
    )


    # Build document context

    context = ""

    for i, result in enumerate(
        results,
        start=1
    ):

        context += (
            f"\n\n"
            f"[Source {i} - Page "
            f"{result['page']}]\n"
            f"{result['text']}"
        )


    # System instructions

    system_prompt = """
You are a PDF question-answering assistant.

Answer the user's question ONLY using the
information contained in the supplied document context.

Do not invent information.

If the answer is not present in the context,
say:

"I could not find this information in the uploaded document."

When possible, mention the PDF page number.

Keep answers clear and concise.
"""


    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


QUESTION:

{question}
"""


    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=1200
    )


    return response.choices[0].message.content


# =========================
# PDF UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# =========================
# PROCESS PDF
# =========================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    file_id = (
        uploaded_file.name,
        len(pdf_bytes)
    )


    # Only process a new PDF

    if st.session_state.get(
        "file_id"
    ) != file_id:

        with st.spinner(
            "Processing PDF..."
        ):

            # Extract text

            pages = extract_pdf(
                pdf_bytes
            )


            if not pages:

                st.error(
                    "No readable text was found "
                    "in this PDF."
                )

                st.stop()


            # Create chunks

            chunks = create_chunks(
                pages
            )


            # Load embedding model

            model = load_embedding_model()


            # Create embeddings

            embeddings = create_embeddings(
                chunks,
                model
            )


            # Create FAISS index

            index = create_faiss_index(
                embeddings
            )


            # Save everything in session

            st.session_state.file_id = file_id

            st.session_state.pages = pages

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.chat_history = []


        st.success(
            f"PDF processed successfully! "
            f"{len(pages)} pages and "
            f"{len(chunks)} chunks created."
        )


# =========================
# DOCUMENT INFORMATION
# =========================

if "chunks" in st.session_state:

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "PDF Pages",
            len(st.session_state.pages)
        )

    with col2:

        st.metric(
            "Text Chunks",
            len(st.session_state.chunks)
        )


# =========================
# CHAT HISTORY
# =========================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================
# USER QUESTION
# =========================

question = st.chat_input(
    "Ask a question about the PDF..."
)


if question:

    # Make sure PDF exists

    if "index" not in st.session_state:

        st.warning(
            "Please upload a PDF first."
        )

        st.stop()


    # Display question

    with st.chat_message("user"):

        st.markdown(
            question
        )


    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Generate answer

    with st.chat_message("assistant"):

        try:

            model = load_embedding_model()


            # Search FAISS

            with st.spinner(
                "Searching the document..."
            ):

                results = search_document(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    model
                )


            # Show retrieved chunks

            with st.expander(
                "🔎 Retrieved sources"
            ):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"**Source {i} — "
                        f"Page {result['page']} — "
                        f"Similarity "
                        f"{result['score']:.3f}**"
                    )

                    st.write(
                        result["text"]
                    )


            # Ask Groq

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    results
                )


            st.markdown(
                answer
            )


            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


        except Exception as error:

            st.error(
                f"Error: {error}"
            )
