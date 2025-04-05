from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
# VECTOR_STORE_ID="vs_67ea76dc84988191b848a57b0f6b7b8e"

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_MESSAGE = r"""You are an assistant, expert in telecommunication systems. Your task is to answer
users' questions using the information provided in the attached file. Always look through the file when answering a
question. You should never try to use your own knowledge before checking the information in the file."""

def assistant_conversation(openaiClient=client):
    while True:
        lines = []
        while True:
            line = input(">> ")
            if line == "END":
                break
            lines.append(line)
        user_input = "\n".join(lines)

        moderation_check = openaiClient.moderations.create(
            model="omni-moderation-latest",
            input=user_input,
        )
        flagged = moderation_check.results[0].flagged
        if flagged:
            print("Your query violates ToS and will not be answered. Please input another query.")
            continue

        response = openaiClient.responses.create(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.15,
            instructions=SYSTEM_MESSAGE,
            input=user_input,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID]
            }]
        )
        response_text = response.output[1].content[0].text.strip()
        print("Raw response text:")
        print(response_text)
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")

if __name__ == "__main__":
    assistant_conversation()