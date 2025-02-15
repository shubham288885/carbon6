from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from typing import List
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from document_processors import load_multimodal_data
from utils import set_environment_variables
import logging
from fastapi.middleware.cors import CORSMiddleware

import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "chat_templates"))
maps1_templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), ""))

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "chat_templates/ChatUI")), name="static")
app.mount("/assets", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "chat_templates/ChatUI/assets")), name="assets")
app.mount("/maps1", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "maps1")), name="maps1")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize settings
def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")

initialize_settings()

# Connect to Milvus Vector Store
vector_store = MilvusVectorStore(host="127.0.0.1", port=19530, dim=1024)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = None  # Will be initialized when documents are uploaded

class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: QueryRequest):
    global index
    if index is None:
        return {"error": "Index is not initialized. Upload documents first."}  
    try:
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=False)
        response = query_engine.query(request.query)
        return {"response": response.response}
    except Exception as e:
        logging.exception("Error during query execution")
        return {"error": str(e)}

'''@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    global index
    set_environment_variables()

    try:
        # Debugging: Print files to see what's being uploaded
        # Debugging: Print file types and filenames to see what's being uploaded
        for file in files:
            logging.info(f"File type: {type(file)}")
            logging.info(f"File filename: {file.filename}")
            logging.info(f"File content type: {file.content_type}")
        logging.info(f"Files prepared: {[file.filename for file in files]}")  # Log filenames

        # Process the uploaded files
        documents = await load_multimodal_data(files)
        
        # Create and store the index
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return {"message": "Documents uploaded and indexed successfully"}
    
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        return {"error": str(e)}'''
#new_upload
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    global index
    set_environment_variables()

    try:
        logging.info(f"Uploading files: {[file.filename for file in files]}")

        # Process the uploaded files
        documents = await load_multimodal_data(files)

        if index is None:
            # Initialize index if it's the first upload
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        else:
            # Append new documents instead of replacing index
            index = index + VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            logging.info(f"Indexed documents: {[doc.metadata['source'] for doc in documents]}")

        logging.info(f"Current index size: {len(index.docstore)}")

        return {"message": "Documents uploaded and indexed successfully"}
    
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        return {"error": str(e)}


'''@app.get("/")
def read_root():
    return {"message": "FastAPI is running on EC2"}'''

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("ChatUI/index.html", {"request": request}) 

#chat_ui
@app.get("/chat_ui", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("ChatUI/index_chat.html", {"request": request})

#map1
@app.get("/maps1", response_class=HTMLResponse)
async def map1(request: Request):
    return maps1_templates.TemplateResponse("maps1/index.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, but restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)