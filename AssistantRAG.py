from chromadb import Settings
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv

from helper_utils import word_wrap
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_MESSAGE = r"""You are an assistant, expert in telecommunication systems. Your task is to answer
users' questions using the information provided in the attached file. Always look through the file when answering a
question. You should never try to use your own knowledge before checking the information in the file."""

SYSTEM_MESSAGE_FOR_RAG = r"""You are a helpful assistant, expert in telecommunication systems. Your users are
asking questions about information contained in a technical specification document for multiple telecom services. 

You will be shown the user's question, and the relevant information from the technical specification document.
Answer the user's question using ONLY this information.

IMPORTANT: 
1. Be comprehensive - make sure to include ALL relevant parts of the procedure, steps, or information requested.
2. If the information appears to be incomplete, mention this in your response.
3. Follow any enumerated lists or procedural steps as they appear in the document.
4. Present your response in a clear, structured format.
"""

collection_name = "etsi_document_2017"

def get_chroma_collection(col_name=collection_name):
    chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
    return chroma_client.get_or_create_collection(col_name)

def embed_pdf(file_path="RAG_files/etsi_pdf.pdf"):
    chroma_collection = get_chroma_collection()

    if chroma_collection.count() == 0:
        reader = PdfReader(file_path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        print(word_wrap(pdf_texts[17]))

        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=3000,
            chunk_overlap=500
        )
        character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10, tokens_per_chunk=256)
        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)
        ids = [str(i) for i in range(len(token_split_texts))]

        chroma_collection.add(ids=ids, documents=token_split_texts)
        chroma_collection.count()

    return chroma_collection

#TODO: write assistant_conversation_without_chroma() and test 1 vs the other

def assistant_conversation(openaiClient=client):
    embeddings_collection = embed_pdf("RAG_files/etsi_pdf.pdf")

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

        results = embeddings_collection.query(query_texts=user_input, n_results=10)
        retrieved_documents = results['documents'][0]
        information = "\n\n".join(retrieved_documents)

        response = openaiClient.responses.create(
            model="gpt-4o",
            temperature=0.15,
            instructions=SYSTEM_MESSAGE_FOR_RAG,
            input=[
                {
                    "role": "user",
                    "content": f"Question: {user_input}. \n Information: {information}",
                }
            ],
        )
        response_text = response.output[0].content[0].text.strip()
        print("Raw response text:")
        print(response_text)
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")

if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
    chroma_client.reset()
    assistant_conversation()