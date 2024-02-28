import os
import sys
import io
import utils as myutils
from flask import Flask, request, jsonify, render_template, Response, abort
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.llms.vertexai import VertexAI
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from getpass import getpass
from langchain.llms import HuggingFacePipeline
import torch
from transformers import pipeline
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
HUGGINGFACEHUB_API_TOKEN = "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"
generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True, max_new_tokens=128)
hf_pipeline = HuggingFacePipeline(pipeline=generate_text)



os.environ['GOOGLE_APPLICATION_CREDENTIALS']="./listenlayerai-13defcf4a9ae.json"
# Load configuration from environment variables
def load_config():
    x_auth = os.environ.get("X_AUTH", "")
    service_account_file = "./listenlayerai-13defcf4a9ae.json"
    project = "listenlayerai"
    dataset = "accountId_8631a2a7_de4e_44e1_a423_a7c2217fb5a6"
    openai_api_key = "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"
    model = "gpt-4"
    top_k = "1000"
    debug = "False"
    verbose = "True"

    return {
        "service_account_file": service_account_file,
        "project": project,
        "dataset": dataset,
        "openai_api_key": openai_api_key,
        "model": model,
        "top_k": top_k,
        "debug": debug,
        "verbose": verbose,
        "x_auth": x_auth
    }

def execute_and_capture_output(func, *args, **kwargs):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_output
    result = None
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    output = redirected_output.getvalue()
    return result, output

# Function to check if a configuration value was provided
def get_env_variable(var_name):
    try:
        return os.environ[var_name]
    except KeyError:
        error_msg = f"The environment variable {var_name} was not provided!"
        raise Exception(error_msg)

config = load_config()
# Set Environment Variable
os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

# sys.stdout = open('output.log', 'w')
# sys.stderr = sys.stdout


def index():
    return render_template('index.html')


def stream():
    def generate():
        log_file = open('output.log', 'r')  # Open a log file for appending

        while True:
            line = log_file.readline()  # Read a line from the log file
            if not line:
                continue
            yield f"data: {line}\n\n"  # Yield the line as a server-sent event (SSE)

    return Response(generate(), mimetype='text/event-stream')
    # x_auth_header = request.headers.get('x-auth')
    # expected_x_auth = config["x_auth"]
    
    # if expected_x_auth and x_auth_header != expected_x_auth:
    #     abort(401)  # Unauthorized
    
def create_sql_database():

    sericeFile = config["service_account_file"]
    encoded_string = myutils.save_file(sericeFile)

    sqlalchemy_url = f'bigquery://{config["project"]}/{config["dataset"]}?credentials_path={encoded_string}'
    print(sqlalchemy_url)
    return SQLDatabase.from_uri(sqlalchemy_url)

def create_language_model():
    temperature = 0
    model = config["model"]
    if model.startswith("gpt"):
        return ChatOpenAI(temperature=temperature, model=model)
    else:
        return OpenAI(temperature=temperature, model=model)

def create_agent_executor(llm, toolkit, verbose, top_k):
    return create_sql_agent(llm=llm, toolkit=toolkit, verbose=verbose, top_k=top_k)

query="what is David's lastName?"
    # Create SQLDatabase and language model instances
db = create_sql_database()
print(db)
# llm = create_language_model()
sericeFile = config["service_account_file"]
encoded_string = myutils.save_file(sericeFile)
repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
hugging_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
    # Create SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=hf_pipeline)

    # Create SQL Agent Executor
agent_executor = create_agent_executor(llm=hf_pipeline, toolkit=toolkit, verbose=config["verbose"], top_k=config["top_k"])

    # Execute query
result, output = execute_and_capture_output(agent_executor.run, query)
print(output)
queryResult = myutils.get_query(output).replace('"', '')
output = myutils.remove_colors(output)

