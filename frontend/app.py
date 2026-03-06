import streamlit as st
import requests
import json
import time

# API Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Medical Graph RAG",
    page_icon="🧬",
    layout="wide",
)

st.title("Edge-Native Medical Graph RAG 🧬")
st.markdown("""
    This prototype demonstrates a privacy-first, hallucination-free RAG system.
    It combines **Dense Vector Search (pgvector)** with **Knowledge Graph Traversal (Apache AGE)**
    and runs entirely locally via **Ollama (Phi-3.5)**.
""")

# Sidebar for Document Upload
with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload a Medical PDF", type=["pdf"])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF (Chunking, Embedding, Graph Construction)..."):
                try:
                    # Upload and Vectorize
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/api/upload", files=files)
                    
                    if response.status_code == 200:
                        doc_data = response.json()
                        st.success(f"Successfully processed: {doc_data['message']}")
                        doc_id = doc_data['document_id']
                        
                        # Generate Knowledge Graph
                        st.info("Generating Knowledge Graph in Apache AGE...")
                        graph_res = requests.post(f"{API_URL}/api/graph/generate/{doc_id}")
                        if graph_res.status_code == 200:
                            g_data = graph_res.json()
                            st.success(f"Graph Built: {g_data['nodes_extracted']} nodes, {g_data['edges_extracted']} edges.")
                        else:
                            st.error(f"Graph error: {graph_res.text}")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please upload a file first.")

st.divider()

# Main area for Querying
st.header("2. Hybrid Knowledge Retrieval & Inference")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a clinical query (e.g., 'What is the patient's WBC count?'):"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Bot response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Retrieving Vectors, Traversing Graph, and Generating SLM Answer locally..."):
            try:
                payload = {"query": query, "top_k": 3}
                res = requests.post(f"{API_URL}/api/query/generate", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer provided.")
                    context_used = data.get("context_used", {})
                    
                    # Display the final answer
                    response_placeholder.markdown(answer)
                    
                    # Store in history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Display Context Sources in an expander
                    with st.expander("🔍 View Retrieved Context (Vector + Graph)"):
                        st.subheader("Extracted Document Excerpts (pgvector)")
                        chunks = context_used.get("chunks", [])
                        for i, chunk in enumerate(chunks):
                            st.markdown(f"**Chunk {i+1}:**\n> {chunk}")
                            
                        st.subheader("Connected Medical Entities (Apache AGE)")
                        entities = context_used.get("graph_entities", [])
                        if entities:
                            for ent in entities:
                                st.markdown(f"- `{ent}`")
                        else:
                            st.markdown("*No additional graph entities found for this query context.*")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is the FastAPI server running?")
