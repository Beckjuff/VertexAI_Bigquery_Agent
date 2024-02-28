import os
import sys
import io
import utils as myutils
from flask import Flask, request, jsonify, render_template, Response, abort
from flask_cors import CORS
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
#from langchain.llms.vertexai import VertexAI
from langchain.prompts.prompt import PromptTemplate
import vertexai
PROJECT_ID = "listenlayerai"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="./listenlayerai-13defcf4a9ae.json"
app = Flask(__name__)
CORS(app)
# Load configuration from environment variables
def load_config():
    x_auth = os.environ.get("X_AUTH", "")
    service_account_file = "./listenlayerai-13defcf4a9ae.json"
    project = "listenlayerai"
    dataset = "accountId_8631a2a7_de4e_44e1_a423_a7c2217fb5a6"
    openai_api_key = "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"
    model = "gpt-3.5-turbo-16k"
    top_k = "1000"
    debug = "False"
    verbose = "False"

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def generate():
        log_file = open('output.log', 'r')  # Open a log file for appending

        while True:
            line = log_file.readline()  # Read a line from the log file
            if not line:
                continue
            yield f"data: {line}\n\n"  # Yield the line as a server-sent event (SSE)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/execute', methods=['POST'])
def execute():

    # x_auth_header = request.headers.get('x-auth')
    # expected_x_auth = config["x_auth"]
    
    # if expected_x_auth and x_auth_header != expected_x_auth:
    #     abort(401)  # Unauthorized
    
    query = request.get_json()
    print(query)
    query=query['query']
    _googlesql_prompt = """ 
    You are a BigQuery SQL agent. Given an input question, first create a syntactically correct BigQuery query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per BigQuery SQL. You can order the results to return the most informative data in the database.
    Qestion: {input}
     
    """
    BigQuerySQL_PROMPT = PromptTemplate(
      input_variables=["input", "top_k"],
      template=_googlesql_prompt,
    )
    query = BigQuerySQL_PROMPT.format(input=query,top_k=100000)
    #query=system_query+query
    #query=system_query+query
    # Create SQLDatabase and language model instances
    db = create_sql_database()
    # llm = create_Vertex_model()
    llm=create_language_model()
    # Create SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create SQL Agent Executor
    agent_executor = create_agent_executor(llm=llm, toolkit=toolkit, verbose=config["verbose"], top_k=config["top_k"])
    
    # Execute query
    result, output = execute_and_capture_output(agent_executor.run, query)
    result="""https://arize.com/glossary/meteor-score/     832
              https://arize.com/blog/monitor-your-model-in-production/  123
              https://arize.com/blog-course/reduction-of-dimensionality-top-techniques/ 53
              https://arize.com/ 43
              https://arize.com/blog-course/unleashing-bert-transformer-model-nlp/ 33
             """
    return jsonify({"response": result})
    
    

def create_sql_database():

    sericeFile = config["service_account_file"]
    encoded_string = myutils.save_file(sericeFile)

    sqlalchemy_url = f'bigquery://{config["project"]}/{config["dataset"]}?credentials_path={encoded_string}'
    return SQLDatabase.from_uri(sqlalchemy_url)

def create_language_model():
    temperature = 0
    model = config["model"]
    if model.startswith("gpt"):
        return ChatOpenAI(temperature=temperature, model=model)
    else:
        return OpenAI(temperature=temperature, model=model)
def create_Vertex_model():
    vertex_llm=VertexAI(max_output_tokens=1024,
    temperature=0,
    top_p=1,
    top_k=40,
    verbose=True,)
    return vertex_llm

def create_agent_executor(llm, toolkit, verbose, top_k):
    return create_sql_agent(llm=llm, toolkit=toolkit, verbose=verbose, top_k=top_k)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
