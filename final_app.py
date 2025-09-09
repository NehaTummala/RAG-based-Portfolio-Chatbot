import os
import streamlit as st
from dotenv import load_dotenv
from utils import load_documents, split_documents, build_vectorstore, build_qa_chain

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="🤖 RAG Portfolio Assistant",
    page_icon="📄",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>📄 RAG-powered Portfolio Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your <b>resume, project documents, or bios</b> and ask questions!</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_files = st.file_uploader(
    "📎 Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload PDFs like resumes, bios, or project reports"
)

if uploaded_files:
    with st.spinner("🔍 Processing and indexing your documents..."):
        all_docs = []
        os.makedirs("data", exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = load_documents(file_path)
            all_docs.extend(docs)

        chunks = split_documents(all_docs)
        vectorstore = build_vectorstore(chunks)
        qa_chain = build_qa_chain(vectorstore)

    st.success("✅ All documents indexed successfully! You can now ask questions.")
    st.markdown("---")

    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question below 👇")

    if query:
        with st.spinner("💡 Thinking..."):
            response = qa_chain.run(query)

        st.markdown("#### ✅ Answer:")
        st.chat_message("assistant").markdown(response)

else:
    st.info("📂 Upload PDF documents to get started.")