from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

CHROMA_PATH = 'chroma_db'
DATA_PATH = 'data.csv'

def load_document(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError('No file Found')
    elif not filepath.endswith('.csv'):
        raise ValueError("Only csv files are expected")
    loader = CSVLoader(file_path=filepath,csv_args={'delimiter':','})
    docs = loader.load()

    basename = os.path.basename(filepath)
    for doc in docs:
        doc.metadata['source'] = basename
    return docs

def split_doc(docs):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
        )
    chunks = spliter.split_documents(docs)
    return chunks

def get_embedding():
    embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    return embedding

db = None
def vector_store(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    global db
    db = Chroma.from_documents(
            embedding=get_embedding(),
            documents=chunks,
            persist_directory=CHROMA_PATH,
        )
    return db

def load_retriver():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding()
    )
    retriver = db.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k':3
        }
    )
    return retriver

def join_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are MediAssist, an expert medical symptom analysis assistant.
Analyze the user's symptoms using ONLY the context provided below.
Do not use any outside knowledge. If symptoms don't match anything in the context, say 'I don't have enough information.'

Context:
{context}

You MUST respond in exactly this structure:

**Possible Conditions:**
For each matching disease from the context, write:
- **[Disease Name]** (Severity: [severity])
  - Why it matches: [which symptoms align]
  - Key risk factors: [relevant risk factors from context]
  - Watch out for: [additional symptoms to monitor]

**Most Likely Condition:**
[Name the single best match and explain why in 2 sentences]

**Urgency:**
[One of: "Monitor at home", "See a doctor within 24-48 hours", "Seek emergency care immediately"]
[One sentence explaining why]

**Recommended Next Steps:**
[2-3 concrete action points the user should take]

IMPORTANT: You MUST always end with exactly this line:
⚠️ This is not a medical diagnosis. Please consult a qualified doctor."""),

    ("human", "{question}"),
])

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=3000,
    temperature=0.2
)

parser =StrOutputParser()
model = ChatHuggingFace(llm=llm)


def build_chain():

        docs = load_document(DATA_PATH)
        chunks = split_doc(docs)
        vector_store(chunks)
        retriver = load_retriver()

        parallel_chain = RunnableParallel({
        'context':retriver | RunnableLambda(join_docs),
        'question':RunnablePassthrough()
        })

        chain = parallel_chain | prompt | model | parser
        return chain
    
if __name__ == '__main__':
     
     chain = build_chain()
     print('MediAssist ready. Type exit to quit. \n')
     while True:
          
        question = input('you: ').strip()

        if question.lower() == 'exit':
             break
        if not question:
             continue

        response = chain.invoke(question)
        print(f'\nMediAssist: {response}\n')