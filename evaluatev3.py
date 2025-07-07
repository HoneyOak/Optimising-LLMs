
import json
import os
import glob
import fitz  # PyMuPDF
import dspy

lm = dspy.LM('ollama_chat/llama3.2:latest', api_base='http://10.20.200.109:11434', api_key='')
dspy.configure(lm=lm)

# Load PDF and extract text
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=600):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

corpus = []
folder_path = "data"

for pdf_file in glob.glob(os.path.join(folder_path, "*.pdf")):
    filename = os.path.basename(pdf_file)
    print(f"Processing {filename}...")
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        corpus.append({
            "source": filename,
            "chunk_index": i,
            "text": chunk
        })

print(f"Loaded {len(corpus)} chunks from {len(glob.glob(os.path.join(folder_path, '*.pdf')))} PDFs.")

from sentence_transformers import SentenceTransformer

# Load an extremely efficient local model for retrieval
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

embedder = dspy.Embedder(model.encode)

search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=[doc["text"] for doc in corpus],
    k=5,  # top-k documents per query
    brute_force_threshold=30_000  # skip FAISS if corpus is small
)
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
rag = RAG()



import random
from typing import Literal
from dspy.datasets import DataLoader
from datasets import load_dataset
import dspy
import litellm
import ujson
# Enable LiteLLM parameter dropping
litellm.drop_params = True

with open("data/finance.json", "r") as f:
    raw_data = json.load(f)

data = [
    dspy.Example(
        question=ex["question"],  # renaming text → question
        response=ex["answer"]
    ).with_inputs("question")
    for ex in raw_data
]

trainset = data

random.Random(0).shuffle(trainset)
print(f"Training dataset prepared with {len(trainset)} examples.")

from dspy.teleprompt import SIMBA
from dspy.evaluate import SemanticF1

# Optimize via BootstrapFinetune
print("Optimizing the model using+ BootstrapFinetune...")
dspy.settings.experimental = True
optimizer = dspy.SIMBA(metric=SemanticF1(decompositional=True), num_threads=24, max_steps=4, num_candidates=4)
optimized = optimizer.compile(rag, trainset=trainset)
print("Optimization completed successfully.")
print("Optimized model:", optimized)

# Save the optimized model
print("Saving the optimized program...")
optimized.save("optimized_program.json")
print("Optimized program saved as 'optimized_program.json'.")

# Run the optimized classifier on a sample input
print("Running the optimized classifier on a sample input...")
result = optimized("Mary has 5 apples. She buys 7 more. How many apples does she have now?")
print("Result:", result)