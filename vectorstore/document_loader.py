from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import fitz
import gpt4all

urls = [
    "https://sbert.net/docs/installation.html",
    "https://www.sbert.net/docs/quickstart.html"
]

pdf_file_paths = ['utils/howto-ipaddress.pdf', 'utils/howto-sorting.pdf']

def read_pdfs(file_paths):
    text = ""
    for file_path in file_paths:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    return text

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

pdf_text = read_pdfs(pdf_file_paths)
pdf_docs = [Document(page_content=pdf_text)]

combined_docs = docs_list + pdf_docs

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(combined_docs)

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
db = FAISS.from_texts([doc.page_content for doc in doc_splits], embeddings)
db.save_local("willinn-langchain")
new_db = FAISS.load_local("willinn-langchain", embeddings, allow_dangerous_deserialization=True)


retriever = new_db.as_retriever(k=3)
