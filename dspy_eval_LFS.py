import dspy
from dspy import LM
from dspy.evaluate import Evaluate
from dspy import ChainOfThought
from typing import List, Optional
import pandas as pd
import textwrap
import os

os.environ["DSP_CACHEBOOL"] = "False"

# Initialize Phoenix for tracing
import phoenix as px
from phoenix.trace import using_project
from phoenix.evals import LiteLLMModel

# Instrumentation for tracing and monitoring
from openinference.instrumentation.dspy import DSPyInstrumentor
from phoenix.otel import register
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = register(endpoint=endpoint, set_global_tracer_provider=False)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

os.environ['OLLAMA_API_BASE'] = "https://localhost:11434/"

from openinference.instrumentation.litellm import LiteLLMInstrumentor
model = LiteLLMModel(model="ollama/llama3.2:3b")


DSPyInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

lm = dspy.LM('ollama_chat/cas/llama-3.2-3b-instruct:latest', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)
dspy.settings.configure(backoff_time=60)

applicant_info = """
Name: John Doe
Age: 35
Annual Income: $75,000
Credit Score: 720
Existing Debts: $20,000 in student loans, $5,000 in credit card debt
Loan Amount Requested: $250,000 for a home mortgage
Employment: Software Engineer at Tech Corp for 5 years
"""

class ZeroShot(dspy.Module):
    """
    You are given a piece of text that contains information about an applicant. 
    Analyze the applicant's financial information and return a risk assessment.
    """
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict("question -> answer", cache = False)

    def forward(self, applicant):
        return self.prog(question="Analyze the applicant's financial information and return a risk assessment. Applicant: " + applicant)
    
with using_project("zero_shot"):
    module = ZeroShot()
    response = module(applicant_info)
    print(f"ZeroShot response:\n {response}")


#Labeled Few Shot
# class RiskAssessment(dspy.Signature):
#     """Analyze the applicant's financial information and return a risk assessment."""
#     question = dspy.InputField()
#     applicant = dspy.InputField()
#     answer = dspy.OutputField(desc="""
#         A thorough risk analysis about the applicant, justifying the assessment 
#         for each of the parameters considered from the applicant
#     """)

# class RiskAssessmentAgent(dspy.Module):
#     def __init__(self):
#         self.question = "Analyze the applicant's financial information and return a risk assessment."
#         self.assess_risk = ChainOfThought(RiskAssessment, n=3)

#     def forward(self, applicant: str):
#         question = self.question
#         applicant = applicant
#         pred = self.assess_risk(question=question, applicant=applicant)
#         return dspy.Prediction(answer=pred.answer)
    
# import json

# # Load the training data
# dataset = json.load(open("/home/temporaryaccess/Code/virtualEnvs/alice-in-wonderland-main/eval/data/training_data.json", "r"))['examples']
# trainset = [
#     dspy.Example(
#         question="Analyze the applicant's financial information and return a risk assessment",
#         applicant=e['applicant'],
#         answer=e['answer']
#     ) for e in dataset
# ]

# from dspy.teleprompt import LabeledFewShot

# # Train
# teleprompter = LabeledFewShot()
# lfs_optimized_advisor = teleprompter.compile(
#     RiskAssessmentAgent(),
#     trainset=trainset[3:]  # Using part of the dataset for training
# )

# with using_project("labeled_few_shot"):
#     response = lfs_optimized_advisor(applicant_info)
#     wrapped_response = textwrap.fill(response.answer, width=70)
#     print(f"LabeledFewShot Optimized response:\n {wrapped_response}")
