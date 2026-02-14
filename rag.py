# rag.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
load_dotenv()

vector_store = None

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

model = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

def process_document(docs):
    global vector_store
    texts = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model
    )

def get_answer(query: str):
    global vector_store

    if vector_store is None:
        return "Please upload a document first."

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    # IMPROVED PROMPT with formatting instructions
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on provided documents.

Context from the documents:
{context}

Question: {query}

Instructions:
- Answer the question directly and naturally
- Use bullet points ONLY when listing multiple related items
- Keep explanations clear and concise
- Don't add meta-commentary like "Context Explanation" or "Unanswered Question"
- Don't repeat the question
- If information is missing, simply say "I don't have that information in the documents"

Answer:
""")

    rag_chain = (
        {
            "context": retriever | RunnableLambda(
                lambda docs: "\n\n".join(d.page_content for d in docs)
            ),
            "query": RunnablePassthrough()
        }
        | prompt
        | model
        | parser
    )

    return rag_chain.invoke(query)
