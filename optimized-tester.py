
import json

import fitz  # PyMuPDF
import dspy

lm = dspy.LM('ollama_chat/llama3.2:latest', api_base='http://10.20.200.109:11434', api_key='')
dspy.configure(lm=lm)

# Load PDF and extract text
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

pdf_text = extract_text_from_pdf("data/HDFC April25.pdf") 

# Chunk the text to use as documents
def chunk_text(text, chunk_size=600):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

corpus = chunk_text(pdf_text)
print(f"Loaded {len(corpus)} chunks from PDF.")

from sentence_transformers import SentenceTransformer

# Load an extremely efficient local model for retrieval
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

embedder = dspy.Embedder(model.encode)

search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=corpus,
    k=5,  # top-k documents per query
    brute_force_threshold=30_000  # skip FAISS if corpus is small
)
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)

old_rag = RAG()
rag = RAG()
rag.load(path="optimized_program.json")

while True:
    query = input("Prompt: ")
    opt_result = rag(query)
    result = old_rag(query)
    print("Unoptimized Result: \n", result)
    print("Optimized Result: \n", opt_result)