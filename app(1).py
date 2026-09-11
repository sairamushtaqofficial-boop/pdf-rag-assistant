```python
import os
import re
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents using "
    "Retrieval-Augmented Generation (RAG)."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# GET GROQ API KEY
# ============================================================

def get_groq_api_key():

    # Streamlit Cloud Secrets
    try:

        api_key = st.secrets["GROQ_API_KEY"]

        if api_key:
            return api_key

    except Exception:
        pass


    # Local environment variable
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key


    return None


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(pdf_file):

    reader = PdfReader(
        BytesIO(pdf_file)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            # Remove unnecessary whitespace
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


# ============================================================
# CREATE CHUNKS
# ============================================================

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


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks, model):

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


# ============================================================
# CREATE FAISS VECTOR DATABASE
# ============================================================

def create_faiss_database(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# SEARCH FAISS
# ============================================================

def search_documents(
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

    number_of_results = min(
        TOP_K,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        number_of_results
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


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    retrieved_documents
):

    api_key = get_groq_api_key()

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it in Streamlit Secrets."
        )


    client = Groq(
        api_key=api_key
    )


    # Build context
    context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += (
            f"\n\n"
            f"[Source {i} - PDF Page "
            f"{document['page']}]\n"
            f"{document['text']}"
        )


    system_prompt = """
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context.

Do not invent information.

If the answer cannot be found in the provided context,
say:

"I could not find this information in the uploaded document."

When possible, mention the relevant PDF page number.

Keep the answer clear and concise.
"""


    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


USER QUESTION:

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


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    file_id = (
        uploaded_file.name,
        len(file_bytes)
    )


    # Process only when a new PDF is uploaded
    if st.session_state.get(
        "file_id"
    ) != file_id:

        with st.spinner(
            "Processing PDF..."
        ):

            # -------------------------
            # Extract text
            # -------------------------

            pages = extract_pdf_text(
                file_bytes
            )


            if not pages:

                st.error(
                    "No readable text was found "
                    "in this PDF."
                )

                st.stop()


            # -------------------------
            # Create chunks
            # -------------------------

            chunks = create_chunks(
                pages
            )


            # -------------------------
            # Load embedding model
            # -------------------------

            model = load_embedding_model()


            # -------------------------
            # Create embeddings
            # -------------------------

            embeddings = create_embeddings(
                chunks,
                model
            )


            # -------------------------
            # Create FAISS database
            # -------------------------

            index = create_faiss_database(
                embeddings
            )


            # -------------------------
            # Store in session
            # -------------------------

            st.session_state.file_id = file_id

            st.session_state.pages = pages

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.chat_history = []


        st.success(
            f"PDF processed successfully! "
            f"Created {len(chunks)} searchable chunks."
        )


# ============================================================
# SHOW DOCUMENT INFORMATION
# ============================================================

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


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# ASK QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about your PDF..."
)


if question:

    if "index" not in st.session_state:

        st.warning(
            "Please upload a PDF first."
        )

        st.stop()


    # -------------------------
    # Show user question
    # -------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------
    # Generate answer
    # -------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the document..."
        ):

            model = load_embedding_model()


            # Retrieve relevant chunks
            retrieved_documents = search_documents(
                question,
                st.session_state.index,
                st.session_state.chunks,
                model
            )


        # Show retrieved sources
        with st.expander(
            "🔎 Retrieved document chunks"
        ):

            for i, document in enumerate(
                retrieved_documents,
                start=1
            ):

                st.markdown(
                    f"**Source {i} — Page "
                    f"{document['page']} — "
                    f"Similarity: "
                    f"{document['score']:.3f}**"
                )

                st.write(
                    document["text"]
                )


        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    retrieved_documents
                )

                st.markdown(answer)


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
```
