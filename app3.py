import os
import streamlit as st
from dotenv import load_dotenv
from utils import load_documents, split_documents, build_vectorstore, build_qa_chain

# Load key
load_dotenv()

# Page config
st.set_page_config(page_title="Neha’s AI Chatbot", page_icon="💁‍♀️", layout="centered")

# Custom styles
st.markdown("""
<style>
html, body {
    background-color: #f9f9fb;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    font-weight: 600;
}
.gradient-text {
    background: linear-gradient(to right, #f06292, #ba68c8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 3em;
    margin-bottom: 0.3em;
}
.subtext {
    text-align: center;
    font-size: 1.1em;
    color: #555;
}
.chat-bubble {
    background: #ffffff;
    padding: 14px 18px;
    border-radius: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.suggestion-button {
    background-color: #e1bee7;
    color: #4a148c;
    border: none;
    border-radius: 20px;
    padding: 10px 20px;
    margin: 5px;
    cursor: pointer;
    font-size: 0.9em;
}
.suggestion-button:hover {
    background-color: #d1c4e9;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=90)
    st.markdown("### 🤖 Meet Neha’s Chatbot")
    st.markdown("Built to understand her resume, bio, and projects.\nAsk anything to get smart AI-powered answers!")
    st.markdown("---")
    st.markdown("👩‍💻 **Created by Neha Tummala**")
    st.caption("✨ LangChain + FAISS + OpenAI + Streamlit")

# Title & intro
st.markdown("<div class='gradient-text'>Neha’s AI Resume Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload Neha’s resume, project files, or bio PDFs. Ask anything. Let AI help you shine ✨</div>", unsafe_allow_html=True)

# Upload
uploaded_files = st.file_uploader("📎 Upload Neha’s PDFs (resume, projects, etc):", type=["pdf"], accept_multiple_files=True)

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Build vectorstore
if uploaded_files:
    with st.spinner("📚 Reading and indexing documents..."):
        all_docs = []
        os.makedirs("data", exist_ok=True)

        for uploaded_file in uploaded_files:
            path = os.path.join("data", uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            docs = load_documents(path)
            all_docs.extend(docs)

        chunks = split_documents(all_docs)
        vectorstore = build_vectorstore(chunks)
        st.session_state.qa_chain = build_qa_chain(vectorstore)

    st.success("✅ Neha's documents are indexed! Ask away.")
    st.markdown("### 💡 Try asking:")
    cols = st.columns(3)
    examples = [
        "What NLP projects has Neha worked on?",
        "Summarize Neha’s resume in 2 lines.",
        "What experience does Neha have with GenAI?",
        "What ML tools has she used?",
        "How much AWS experience does Neha have?",
        "Does Neha know FastAPI or LangChain?"
    ]
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"ex{i}", use_container_width=True):
                st.session_state.last_query = example

    query = st.chat_input("Ask something about Neha’s work...")
    if query or st.session_state.get("last_query"):
        q = query or st.session_state.pop("last_query")
        with st.spinner("🤔 Thinking..."):
            try:
                answer = st.session_state.qa_chain.run(q)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.session_state.history.append((q, answer))
        st.markdown(f"**💬 You asked:** {q}")
        st.markdown(f"<div class='chat-bubble'>{answer}</div>", unsafe_allow_html=True)

        if st.button("⬇️ Download Chat Log"):
            chat_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.history])
            st.download_button("Save chat", chat_text, file_name="neha_chat_history.txt", mime="text/plain")

else:
    st.info("📂 Upload Neha’s resume or project PDFs to begin.")



