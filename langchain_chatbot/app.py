# app.py
import os
import gradio as gr
from fastapi import FastAPI
import uvicorn
import threading

# ----------------------------
# CRITICAL: Create FastAPI app FIRST (opens port immediately)
# ----------------------------
app = FastAPI()

# Global variables for models (loaded in background)
rag_chain = None
is_ready = False
loading_error = None

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if is_ready else "loading",
        "ready": is_ready,
        "error": str(loading_error) if loading_error else None
    }

# ----------------------------
# Background Model Loading Function
# ----------------------------
def load_models():
    """Load all heavy models in background after server starts"""
    global rag_chain, is_ready, loading_error
    
    try:
        print("🔄 Loading models in background...")
        
        from transformers import pipeline
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.llms import HuggingFacePipeline
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        # STEP 1: Load and Chunk Data
        print("📄 Loading sample data...")
        os.makedirs("data", exist_ok=True)
        sample_path = "data/sample.txt"
        if not os.path.exists(sample_path):
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("This is a sample document for your LangChain chatbot. "
                        "You can add more content here about your specific domain.")
        
        with open(sample_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # STEP 2: Create Embeddings and Vector Store
        print("🧠 Loading embeddings model (this may take 30-60 seconds)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.from_texts(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        
        # STEP 3: Initialize Local LLM
        print("🤖 Loading language model...")
        llm_pipeline = pipeline(
            "text-generation", 
            model="gpt2",
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            device=-1  # Force CPU
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        
        # STEP 4: Create RAG Chain
        print("⛓️ Building RAG chain...")
        template = """Use the following context to answer the question. If you cannot answer based on the context, say so.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        is_ready = True
        print("✅ Models loaded! Chatbot is ready.")
        
    except Exception as e:
        loading_error = e
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()

# ----------------------------
# Chat Function
# ----------------------------
def chatbot(query):
    if not is_ready:
        if loading_error:
            return f"❌ Error loading models: {str(loading_error)}"
        return "⏳ Models are still loading, please wait 30-60 seconds and try again..."
    
    try:
        response = rag_chain.invoke(query)
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# Gradio Interface
# ----------------------------
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything..."),
    outputs=gr.Textbox(lines=5, label="Response"),
    title="🤖 LangChain Local Chatbot",
    description="Chat with a local LLM using LangChain 1.0+, FAISS, and HuggingFace.",
    examples=[
        ["What is this chatbot about?"],
        ["Tell me about the sample document"],
    ]
)

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

# ----------------------------
# Main: Start server immediately, load models in background
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    # Start model loading in background thread
    model_thread = threading.Thread(target=load_models, daemon=True)
    model_thread.start()
    
    print(f"🚀 Server starting on 0.0.0.0:{port}")
    print("⏳ Models will load in background (30-60 seconds)")
    
    # Start server immediately (port opens right away)
    uvicorn.run(app, host="0.0.0.0", port=port)
