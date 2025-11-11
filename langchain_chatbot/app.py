# app.py
import os
import gradio as gr
from fastapi import FastAPI
import uvicorn
import time

# ----------------------------
# CRITICAL: Create FastAPI app FIRST
# ----------------------------
app = FastAPI()

# Global variables for models
rag_chain = None
is_ready = False
loading_error = None
loading_lock = False  # Prevent multiple simultaneous loads

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if is_ready else "not_ready",
        "ready": is_ready,
        "error": str(loading_error) if loading_error else None
    }

@app.get("/test")
def test_endpoint():
    """Simple test endpoint to verify server is responding"""
    print("🧪 TEST ENDPOINT CALLED")
    return {"message": "Server is responding!", "timestamp": time.time()}

# ----------------------------
# Lazy Model Loading (on first request)
# ----------------------------
def load_models():
    """Load all models - called on FIRST request only"""
    global rag_chain, is_ready, loading_error, loading_lock
    
    # Prevent multiple simultaneous loads
    if loading_lock or is_ready:
        return
    
    loading_lock = True
    start_time = time.time()
    
    try:
        print("\n" + "="*60)
        print("🔄 STARTING MODEL LOADING...")
        print("="*60)
        
        from transformers import pipeline
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.llms import HuggingFacePipeline
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        # STEP 1: Load and Chunk Data
        print("📄 Step 1/4: Loading sample data...")
        os.makedirs("data", exist_ok=True)
        sample_path = "data/sample.txt"
        if not os.path.exists(sample_path):
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("""This is a sample document for your LangChain chatbot.

LangChain Framework:
LangChain is a framework for building applications with large language models. It provides tools for document loading, text splitting, embeddings, vector stores, and chains that connect different components together.

Embeddings:
Embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings, which allows for semantic search.

Vector Stores:
Vector stores like FAISS enable fast similarity search over embeddings. They index document chunks and retrieve the most relevant ones for a given query.

RAG (Retrieval-Augmented Generation):
RAG combines retrieval and generation. It retrieves relevant context from documents and uses that context to generate accurate, grounded responses.""")
        
        with open(sample_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # STEP 2: Create Embeddings and Vector Store
        print("🧠 Step 2/4: Loading embeddings model...")
        print("   (This downloads ~80MB on first run, please wait...)")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("   ✓ Embeddings model loaded")
        
        print("🔨 Step 3/4: Creating vector store...")
        vector_store = FAISS.from_texts(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        print("   ✓ Vector store created")
        
        # STEP 3: Initialize Local LLM
        print("🤖 Step 4/4: Loading GPT-2 language model...")
        llm_pipeline = pipeline(
            "text-generation", 
            model="gpt2",
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            device=-1  # Force CPU
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        print("   ✓ Language model loaded")
        
        # STEP 4: Create RAG Chain
        print("⛓️  Building RAG chain...")
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
        
        elapsed = time.time() - start_time
        is_ready = True
        
        print("="*60)
        print(f"✅ SUCCESS! All models loaded in {elapsed:.1f} seconds")
        print("="*60)
        print("🎉 CHATBOT IS NOW READY TO USE!")
        print("="*60 + "\n")
        
    except Exception as e:
        loading_error = e
        print("\n" + "="*60)
        print(f"❌ ERROR LOADING MODELS:")
        print("="*60)
        print(str(e))
        print("="*60)
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
    finally:
        loading_lock = False

# ----------------------------
# Chat Function
# ----------------------------
def chatbot(query):
    """Handle chatbot queries - loads models on first request"""
    global is_ready, rag_chain
    
    # LOG EVERY REQUEST
    print(f"\n{'='*60}")
    print(f"🔵 CHATBOT FUNCTION CALLED")
    print(f"📥 Query received: '{query}'")
    print(f"📊 Status - Ready: {is_ready}, Loading: {loading_lock}, Error: {loading_error is not None}")
    print(f"{'='*60}")
    
    # Load models on first request (lazy loading)
    if not is_ready and not loading_lock and not loading_error:
        print("\n⚡ First query received, loading models now...")
        load_models()
    
    # Check status
    if loading_error:
        return f"""❌ **Error loading models:**

{str(loading_error)}

Please check the logs or contact support."""
    
    if not is_ready:
        return """⏳ **Models are loading right now...**

This is your FIRST query, so models are being downloaded and loaded.
This takes 60-90 seconds on the first request.

Please wait about 1 minute and submit your question again."""
    
    if not query or query.strip() == "":
        return "Please enter a question."
    
    try:
        print(f"📝 Processing query: '{query[:50]}...'")
        response = rag_chain.invoke(query)
        
        # Clean up the response
        cleaned = response.strip()
        
        # GPT-2 sometimes repeats the prompt
        if query in cleaned:
            cleaned = cleaned.replace(query, "").strip()
        
        # Remove "Answer:" prefix if present
        if cleaned.lower().startswith("answer:"):
            cleaned = cleaned[7:].strip()
        
        return cleaned if cleaned else "I couldn't generate a proper response. Please try rephrasing your question."
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing your request: {str(e)}"

# ----------------------------
# Gradio Interface
# ----------------------------
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Ask me anything about LangChain, embeddings, or RAG...",
        label="Your Question"
    ),
    outputs=gr.Textbox(
        lines=8, 
        label="Response",
        show_copy_button=True
    ),
    title="🤖 LangChain Local Chatbot",
    description="""**Note:** Models load on your FIRST query (takes ~60 seconds). Subsequent queries are instant!

Chat with a local LLM using LangChain, FAISS, and HuggingFace.""",
    examples=[
        ["What is this chatbot about?"],
        ["Tell me about LangChain"],
        ["What are embeddings?"],
        ["Explain RAG"],
    ],
    theme="default",
    flagging_mode="never"
)

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

# ----------------------------
# Main: Start server
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    print("\n" + "="*60)
    print("🚀 LANGCHAIN CHATBOT SERVER")
    print("="*60)
    print(f"🌐 Starting server on 0.0.0.0:{port}")
    print("💡 Models will load on FIRST user query (lazy loading)")
    print("="*60 + "\n")
    
    # Start server immediately
    uvicorn.run(app, host="0.0.0.0", port=port)
