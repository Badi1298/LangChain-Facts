from dotenv import load_dotenv

import langchain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

langchain.debug = True

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory='emb',
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=chat,
)

result = chain.run('What is an interesting fact about the Enlish language?')

print(result)