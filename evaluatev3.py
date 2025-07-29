
import json
import os
import glob
import fitz  # PyMuPDF
import dspy
import random 
lm = dspy.LM('ollama_chat/llama3.2:latest', api_base='http://10.20.200.144:11434', api_key='')
# lm = dspy.LM('ollama_chat/llama3.2:latest', api_base='http://127.0.0.1:11434')
dspy.configure(lm=lm)




from sentence_transformers import SentenceTransformer

# Load an extremely efficient local model for retrieval
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

embedder = dspy.Embedder(model.encode)

with open('data/finance.json', 'r') as f:
    corpus_data = json.load(f)

# Extract the "context" field from each item in the corpus
context_corpus = [doc["context"] for doc in corpus_data]
search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=context_corpus,
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
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.ensemble import Ensemble
# Enable LiteLLM parameter dropping
litellm.drop_params = True

with open("data/finance.json", "r") as f:
    raw_data = json.load(f)

data = [
    dspy.Example(
        question=ex["question"],
        response=ex["answer"]
    ).with_inputs("question")
    for ex in raw_data
]

trainset = data

random.Random(0).shuffle(trainset)
print(f"Training dataset prepared with {len(trainset)} examples.")

from dspy.teleprompt import SIMBA
from dspy.evaluate import SemanticF1

# Optimize
print("Optimizing the model using ...")
dspy.settings.experimental = True
optimizer = dspy.MIPROv2(metric=SemanticF1(decompositional=True), prompt_model= lm, max_bootstrapped_demos=4, auto= "light", num_threads= 24, verbose= True)
optimized = optimizer.compile(rag, trainset=trainset, requires_permission_to_run=False)



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