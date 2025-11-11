# app.py
import os
import gradio as gr
from fastapi import FastAPI
import uvicorn
import threading
import time

# ----------------------------
# CRITICAL: Create FastAPI app FIRST
# ----------------------------
app = FastAPI()

# Global variables for models
rag_chain = None
is_ready = False
loading_error = None
loading_start_time = None

@app.get("/health")
def health_check():
    elapsed = time.time() - loading_start_time if loading_start_time else 0
    return {
        "status": "healthy" if is_ready else "loading",
        "ready": is_ready,
        "loading_time_seconds": int(elapsed),
        "error": str(loading_error) if loading_error else None
    }

# ----------------------------
# Background Model Loading Function
# ----------------------------
def load_models():
    """Load all heavy models in background after server starts"""
    global rag_chain, is_ready, loading_error, loading_start_time
    
    loading_start_time = time.time()
    
    try:
        print("🔄 Loading models in background...")
        
        from transformers import pipeline
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
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
                        "You can add more content here about your specific domain. "
                        "LangChain is a framework for building applications with large language models. "
                        "It provides tools for document loading, text splitting, embeddings, vector stores, "
                        "and chains that connect different components together.")
        
        with open(sample_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        # STEP 2: Create Embeddings and Vector Store
        print("🧠 Loading embeddings model (downloading if needed, ~30-45 seconds)...")
        
        # Cache directory for Hugging Face models
        cache_dir = "/opt/render/project/.cache/huggingface"
        os.makedirs(cache_dir, exist_ok=True)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder=cache_dir
        )
        print("✅ Embeddings model loaded")
        
        # Check if FAISS index exists
        faiss_path = "faiss_index"
        if os.path.exists(faiss_path):
            print("📦 Loading existing FAISS index from disk...")
            vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print("🔨 Creating new FAISS index...")
            vector_store = FAISS.from_texts(chunks, embeddings)
            vector_store.save_local(faiss_path)
            print("💾 FAISS index saved to disk")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        print("✅ Vector store ready")
        
        # STEP 3: Initialize Local LLM
        print("🤖 Loading language model (GPT-2)...")
        llm_pipeline = pipeline(
            "text-generation", 
            model="gpt2",
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            device=-1,  # Force CPU
            model_kwargs={"cache_dir": cache_dir}
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        print("✅ Language model loaded")
        
        # STEP 4: Create RAG Chain
        print("⛓️ Building RAG chain...")
        template = """You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer: """
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        elapsed = time.time() - loading_start_time
        is_ready = True
        print(f"✅ ALL MODELS LOADED! Ready in {elapsed:.1f} seconds")
        
    except Exception as e:
        loading_error = e
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()

# ----------------------------
# Chat Function
# ----------------------------
def chatbot(query):
    """Handle chatbot queries with better status messages"""
    
    if loading_error:
        return f"❌ **Error loading models:** {str(loading_error)}\n\nPlease refresh the page or contact support."
    
    if not is_ready:
        elapsed = time.time() - loading_start_time if loading_start_time else 0
        return f"⏳ **Models are still loading...** ({int(elapsed)}s elapsed)\n\nPlease wait 30-60 seconds total and try again. The first load takes longer as models are downloaded."
    
    if not query or query.strip() == "":
        return "Please enter a question."
    
    try:
        print(f"📝 Processing query: {query[:50]}...")
        response = rag_chain.invoke(query)
        
        # Clean up the response
        cleaned = response.strip()
        
        # GPT-2 sometimes repeats the prompt, remove it
        if query in cleaned:
            cleaned = cleaned.replace(query, "").strip()
        
        # Remove "Answer:" prefix if present
        if cleaned.lower().startswith("answer:"):
            cleaned = cleaned[7:].strip()
        
        return cleaned if cleaned else "I couldn't generate a proper response. Please try rephrasing your question."
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return f"Error processing your request: {str(e)}"

# ----------------------------
# Gradio Interface
# ----------------------------
def get_interface_description():
    """Dynamic description based on loading state"""
    if loading_error:
        return "⚠️ **Error:** Models failed to load. Please refresh or contact support."
    elif not is_ready:
        return "⏳ **Loading models...** This takes 30-60 seconds on first startup. Please wait before asking questions."
    else:
        return "✅ **Ready!** Chat with a local LLM using LangChain, FAISS, and HuggingFace."

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Type your question here... (wait for models to load first)",
        label="Your Question"
    ),
    outputs=gr.Textbox(
        lines=8, 
        label="Response",
        show_copy_button=True
    ),
    title="🤖 LangChain Local Chatbot",
    description=get_interface_description(),
    examples=[
        ["What is this chatbot about?"],
        ["Tell me about LangChain"],
        ["What are embeddings?"],
    ],
    theme="default",
    allow_flagging="never"
)

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

# ----------------------------
# Main: Start server immediately, load models in background
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    print("="*60)
    print("🚀 STARTING LANGCHAIN CHATBOT")
    print("="*60)
    
    # Start model loading in background thread
    model_thread = threading.Thread(target=load_models, daemon=True)
    model_thread.start()
    
    print(f"🌐 Server starting on 0.0.0.0:{port}")
    print("⏳ Models will load in background (30-60 seconds)")
    print("="*60)
    
    # Start server immediately (port opens right away)
    uvicorn.run(app, host="0.0.0.0", port=port)
