from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from rag_chain import build_chain

chain = None

class Query(BaseModel):
    question : str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    chain = build_chain()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/')
def health():
    return {'status': 'ok','Message':'Mediassist is Running'}

@app.post('/ask')
def ask(query: Query):
    response = chain.invoke(query.question)
    return{'answer':response}