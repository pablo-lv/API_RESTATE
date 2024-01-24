import os
import json

import openai
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
print(openai.api_key)


LLM_MODEL = "gpt-3.5-turbo-instruct"
def query_agent(data, query):
    df = pd.read_json(data)
    llm = OpenAI(model_name=LLM_MODEL)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    return agent.run(query)

data = '[{"year": "2018", "price": "100"}, {"year": "2019", "price": "200"}, {"year": "2020", "price": "300"}, {"year": "2021", "price": "400"}, {"year": "2022", "price": "500"}, {"year": "2023", "price": "600"}]'

query = """
Based on the columns years, and price 
Estimate the price for the years 2024, 2025, 2026
And return it in JSON FORMAT with keys YEAR and price
"""

answer = query_agent(data, query)

print(json.dumps(answer))
