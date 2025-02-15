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

from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlas
from langchain.embeddings import OpenAIEmbeddings

# MongoDB Atlas Connection
MONGO_URI = os.getenv("MONGO_URI", "your-mongodb-atlas-uri")
client = MongoClient(MONGO_URI)
db = client["your_database_name"]
collection = db["your_collection_name"]

# Initialize MongoDB Atlas Vector Store
embedding_model = OpenAIEmbeddings()
vector_store = MongoDBAtlas(collection, embedding_model)

class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        results = vector_store.similarity_search(request.query, k=5)  # Retrieve top 5 similar documents
        if not results:
            return {"response": "No relevant context found."}

        response_text = results[0].page_content  # Extract best match
        return {"response": response_text}

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
    set_environment_variables()

    try:
        logging.info(f"Uploading files: {[file.filename for file in files]}")

        # Process files into documents
        documents = await load_multimodal_data(files)

        # Store documents in MongoDB Atlas Vector Store
        vector_store.add_documents(documents)

        logging.info(f"Documents stored in MongoDB Atlas: {[doc.metadata['source'] for doc in documents]}")
        return {"message": "Documents uploaded and indexed successfully in MongoDB Atlas"}

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
