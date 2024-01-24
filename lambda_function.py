import os
import json
from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize OpenAI model
LLM_MODEL = "gpt-3.5-turbo-instruct"
llm = OpenAI(model_name=LLM_MODEL)

def lambda_handler(event, context):
    # Extract the query from the Lambda event
    query = event.get('query', '')

    # Check if the query is empty
    if not query:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Query parameter is missing'})
        }

    # Invoke the OpenAI model with the provided query
    response = llm.invoke(query)

    # Log the response (optional)
    print(json.dumps(response))

    # Return the response
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
