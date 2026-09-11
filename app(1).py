import os
import re
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_pages(pdf_bytes: bytes):
    """Extract text page-by-page so retrieved chunks can cite page numbers."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Character-based chunking with overlap."""
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks


def build_chunks(pages):
    """Create chunks while preserving their PDF page numbers."""
    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk in page_chunks:
            all_chunks.append(
                {
                    "text": chunk,
                    "page": page["page"],
                }
            )

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(chunks, embedding_model):
    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_index(query, index, chunks, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


# -----------------------------
# Groq answer generation
# -----------------------------
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    # Streamlit Cloud/local .streamlit/secrets.toml
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return Groq(api_key=api_key)


def generate_answer(question, retrieved_chunks):
    client = get_groq_client()

    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a document question-answering assistant.

Answer the user's question using ONLY the supplied document context.
Do not invent facts that are not supported by the context.

Rules:
1. If the answer is not present in the retrieved context, say:
   "I could not find this information in the uploaded document."
2. Give a clear, direct answer.
3. When useful, cite the PDF page number in the form [Page X].
4. Do not mention internal RAG, vector databases, embeddings, or retrieval
   unless the user explicitly asks about how the system works.
"""

    user_prompt = f"""Document context:

{context}

User question:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF → extract text → chunk → embed with an open-source "
    "embedding model → search with FAISS → answer with an open-weight model via Groq."
)

with st.sidebar:
    st.header("Settings")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Chunk size:** {CHUNK_SIZE} characters")
    st.write(f"**Chunk overlap:** {CHUNK_OVERLAP} characters")
    st.write(f"**Retrieved chunks:** {TOP_K}")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="The PDF is processed in the current Streamlit session.",
)

if uploaded_file is None:
    st.info("Upload a PDF to build its searchable knowledge base.")
    st.stop()

# Rebuild only when a new file is uploaded.
file_bytes = uploaded_file.getvalue()
file_signature = (uploaded_file.name, len(file_bytes))

if st.session_state.get("file_signature") != file_signature:
    with st.spinner("Processing PDF..."):
        pages = extract_pdf_pages(file_bytes)

        if not pages:
            st.error(
                "No extractable text was found. This app currently works best "
                "with text-based PDFs. Scanned/image-only PDFs need OCR."
            )
            st.stop()

        chunks = build_chunks(pages)

        if not chunks:
            st.error("No text chunks could be created from this PDF.")
            st.stop()

        embedding_model = load_embedding_model()
        index = create_faiss_index(chunks, embedding_model)

        st.session_state.file_signature = file_signature
        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.index = index

        # Reset chat when a new PDF is uploaded.
        st.session_state.messages = []

st.success(
    f"Indexed **{len(st.session_state.pages)} pages** into "
    f"**{len(st.session_state.chunks)} chunks**."
)

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the uploaded PDF...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            embedding_model = load_embedding_model()

            retrieved = search_index(
                question,
                st.session_state.index,
                st.session_state.chunks,
                embedding_model,
                top_k=TOP_K,
            )

            with st.expander("Retrieved context"):
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Source {i} — Page {item['page']} — "
                        f"Similarity: {item['score']:.3f}**"
                    )
                    st.write(item["text"])

            answer = generate_answer(question, retrieved)
            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as exc:
            error_message = f"Error: {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
