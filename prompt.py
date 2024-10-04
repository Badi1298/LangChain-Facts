from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory='emb',
    embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    chain_type="stuff",
    llm=chat,
)

result = chain.run('What is an interesting fact about the Enlish language?')

print(result)