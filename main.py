# main.py

from fastapi import FastAPI, UploadFile, File, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from rag import process_document, get_answer

app = FastAPI()

# In-memory session storage
chat_sessions = {}

class QueryRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/session")
def create_session(response: Response):
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    response.set_cookie(key="session_id", value=session_id)
    return {"session_id": session_id}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    process_document(docs)

    return {"message": "Document processed successfully"}


@app.post("/ask")
def ask_question(data: QueryRequest, request: Request):
    session_id = request.cookies.get("session_id")

    if not session_id:
        return {"error": "Session not found. Refresh page."}

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({
        "role": "user",
        "message": data.question
    })

    answer = get_answer(data.question)

    chat_sessions[session_id].append({
        "role": "bot",
        "message": answer
    })

    return {"answer": answer}
