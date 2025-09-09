import os
import streamlit as st
from dotenv import load_dotenv
from utils import load_documents, split_documents, build_vectorstore, build_qa_chain

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Portfolio Assistant", layout="centered")
st.title("RAG-powered Personal Portfolio Assistant")
st.write("Upload your **resumes, bios, or project documents** (PDFs) and ask questions!")

# File uploader: multiple=True
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔍 Processing and indexing your documents..."):
        all_docs = []
        os.makedirs("data", exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load this PDF and add to master list
            docs = load_documents(file_path)
            all_docs.extend(docs)

        # Step 2: Split all docs into chunks
        chunks = split_documents(all_docs)

        # Step 3: Embed and build vectorstore
        vectorstore = build_vectorstore(chunks)

        # Step 4: Build QA chain
        qa_chain = build_qa_chain(vectorstore)

    st.success("All documents indexed successfully. You can now ask questions!")

    # Input field
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Generating response..."):
            response = qa_chain.run(query)
            st.markdown("**Answer:**")
            st.write(response)
