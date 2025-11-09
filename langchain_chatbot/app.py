# app.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import gradio as gr
import os

# ----------------------------
# STEP 1: Load and Chunk Your Data
# ----------------------------
with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
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
    answer = result["result"]
    return answer

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
iface.launch()



