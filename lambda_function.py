import os
import json

import openai
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI

openai.api_key = os.environ['OPENAI_API_KEY']
print("OPENAI API KEY LOADED")

LLM_MODEL = "gpt-3.5-turbo-instruct"

def query_agent(data, query):
    df = pd.read_json(data)
    llm = OpenAI(model_name=LLM_MODEL)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    return agent.run(query)

def lambda_handler(event, context):
    # Check if 'body' is present (for API Gateway)
    if 'body' in event:
        data = json.loads(event['body'])
    else:
        # If 'body' is not present, assume the event is the data
        data = event

    query = """
    Based on the columns years, and price 
    Estimate the price for the years 2024, 2025, 2026
    And return it in JSON FORMAT with keys YEAR and price
    """

    # Call the query_agent function with the provided data
    answer = query_agent(json.dumps(data), query)

    # Return the response
    response = {
        'statusCode': 200,
        'body': json.dumps(answer)
    }

    return response
