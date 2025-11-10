import os
import gradio as gr
from transformers import pipeline

# --- LangChain imports (2025+ structure) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.llms import HuggingFacePipeline

# ----------------------------
# STEP 1: Load and Chunk Your Data
# ----------------------------
os.makedirs("data", exist_ok=True)
sample_path = "data/sample.txt"

if not os.path.exists(sample_path):
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("This is a sample document for your LangChain chatbot.")

with open(sample_path, "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_text(text)

# ----------------------------
# STEP 2: Create Embeddings and Vector Store
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embeddings)

# ----------------------------
# STEP 3: Initialize Local LLM
# ----------------------------
llm_pipeline = pipeline("text-generation", model="gpt2", max_length=256, temperature=0.7)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ----------------------------
# STEP 4: Create Retrieval QA Chain
# ----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# ----------------------------
# STEP 5: Define Chat Function
# ----------------------------
def chatbot(query):
    result = qa_chain({"query": query})
    return result["result"]

# ----------------------------
# STEP 6: Launch Gradio Interface
# ----------------------------
iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="LangChain Local Chatbot",
    description="Chat with a local LLM using LangChain, FAISS, and HuggingFace."
)

iface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 8080))
)


