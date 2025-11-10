import importlib
import os
import gradio as gr
from transformers import pipeline

def dynamic_import(module_paths, class_name):
    for path in module_paths:
        try:
            module = importlib.import_module(path)
            return getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError):
            continue
    raise ImportError(f"Could not find {class_name} in any of {module_paths}")

# Dynamic LangChain imports
RecursiveCharacterTextSplitter = dynamic_import(
    ["langchain.text_splitter", "langchain_text_splitters"],
    "RecursiveCharacterTextSplitter"
)

RetrievalQA = dynamic_import(
    ["langchain.chains", "langchain.chains.question_answering", "langchain.chains.qa"],
    "RetrievalQA"
)

HuggingFacePipeline = dynamic_import(
    ["langchain.llms", "langchain.llms.huggingface_pipeline"],
    "HuggingFacePipeline"
)

FAISS = dynamic_import(
    ["langchain.vectorstores", "langchain_community.vectorstores"],
    "FAISS"
)

HuggingFaceEmbeddings = dynamic_import(
    ["langchain.embeddings", "langchain_community.embeddings"],
    "HuggingFaceEmbeddings"
)

# Load and split data
with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_text(text)

# Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embeddings)

# Initialize LLM
llm_pipeline = pipeline("text-generation", model="gpt2", max_length=256, temperature=0.7)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Create QA chain
try:
    # Try from_chain_type (newer versions)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
except AttributeError:
    # Older versions fallback
    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )

# Chat function
def chatbot(query):
    result = qa_chain({"query": query})
    return result["result"]

# Launch Gradio
iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="LangChain Local Chatbot",
    description="Chat with a local LLM using LangChain, FAISS, and HuggingFace."
)
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
