import streamlit as st
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


st.set_page_config(page_title="Custom RAG Agent", page_icon="🤖")


# ---------- API KEY ----------

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🤖 Custom RAG Agent")
    key = st.text_input(
        "🔐 Enter OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    if st.button("🚀 Continue", use_container_width=True):

        if key.strip():
            st.session_state.api_key = key.strip()
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Please enter your OpenAI API key.")

    st.stop()


# ---------- SESSION ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None


# ---------- SIDEBAR ----------

st.title("Custom RAG Agent 🤖")

with st.sidebar:

    st.header("⚙️ RAG Settings")

    file = st.file_uploader("📄 Upload PDF", type=["pdf"])

    chunk_size = st.slider(
        "Chunk Size", 200, 2000, 1000, 100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap", 0, 500, 200, 50
    )

    top_k = st.slider(
        "Retrieved Chunks", 1, 10, 3
    )

    process = st.button(
        "🚀 Process Document",
        use_container_width=True
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ---------- PROCESS PDF ----------

if process:

    if not file:
        st.error("Please upload a PDF.")

    else:

        with st.spinner("Processing document..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as f:

                f.write(file.getvalue())
                path = f.name

            docs = PyPDFLoader(path).load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=st.session_state.api_key
            )

            st.session_state.db = FAISS.from_documents(
                chunks, embeddings
            )

            st.session_state.document = file.name
            st.session_state.top_k = top_k
            st.session_state.messages = []

        st.success(f"Document processed! {len(chunks)} chunks created.")


# ---------- CHAT HISTORY ----------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------- CHAT ----------

if st.session_state.db:

    st.info(f"📄 Document: {st.session_state.document}")

    question = st.chat_input(
        "Ask something about your document..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Previous conversation
        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages[:-1]
        )

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=st.session_state.api_key
        )

        # Convert follow-up question into search query
        search = llm.invoke(f"""
Convert the latest question into a standalone
search query using the conversation if necessary.

Conversation:
{history}

Question:
{question}

Return only the search query.
""").content.strip()

        # Search PDF
        docs = st.session_state.db.similarity_search(
            search,
            k=st.session_state.top_k
        )

        context = "\n\n".join(
            d.page_content for d in docs
        )

        # Answer
        answer = llm.invoke(f"""
You are a document-based RAG assistant.

Answer using ONLY the document context.
Use the conversation to understand follow-up questions.
Do not invent information.

If the answer is not in the document, say:
"I couldn't find this information in the document."

Conversation:
{history}

Document:
{context}

Question:
{question}
""").content

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.info("👈 Upload a PDF and click Process Document to start.")