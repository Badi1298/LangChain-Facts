from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores.chroma import Chroma

load_dotenv()

loader = TextLoader('facts.txt')

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

embeddings = OpenAIEmbeddings()

docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory='emb'
)

results = db.similarity_search("What is an interesting fact about the English language?")

for result in results:
    print("\n")
    print(result.page_content)
    
