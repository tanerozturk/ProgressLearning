import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

def auth():
    """ Load environment variables from a .env file located in the same directory.
        Your .env file should contain: GOOGLE_API_KEY="AIzaSy..."
    """
    load_dotenv()

    my_api_key = os.getenv("GOOGLE_API_KEY")

    if not my_api_key:
        print("Error: GOOGLE_API_KEY not found in .env")
        exit()

    return my_api_key

def query(client):
    prompt = input("Enter prompt: ")

    config = types.GenerateContentConfig(
        max_output_tokens=800,
        temperature=0.3,
        top_k = 40,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config
    )

    return response

def show_response(response):
    print('-'*20)
    print(response.text)
    print('-'*20)

def main():
    my_api_key = auth()

    print("Get API key ...")
    client = genai.Client(api_key=my_api_key)

    print("Sending request to Google...")
    response = query(client)

    print("Response received:")
    show_response(response)

if __name__ == "__main__":
    MODEL_NAME = "gemini-2.5-flash"
    main()
