from venv import create

from chromadb import Settings
from openai import OpenAI, embeddings
import chromadb
import os
from dotenv import load_dotenv
from sympy import textplot

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

collection_name = "etsi_document_2017_OPENAI-VER"

def create_openai_embeddings(text, client=client):
    embeddings = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        encoding_format="float"
    ).data[0].embedding
    return embeddings

def get_chroma_collection(col_name=collection_name):
    chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
    return chroma_client.get_or_create_collection(col_name)

def embed_pdf(file_path="RAG_files/etsi_pdf.pdf"):
    chroma_collection = get_chroma_collection()

    if chroma_collection.count() == 0:
        reader = PdfReader(file_path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]

        embeddings = []
        documents = []
        ids = []
        metadatas = []

        for i, page_text in enumerate(pdf_texts):
            if page_text:  # Skip empty pages
                embedding = create_openai_embeddings(page_text)
                embeddings.append(embedding)
                documents.append(page_text)
                ids.append(f"page_{i}")
                metadatas.append({"source": file_path, "page": i})

        # Add to collection
        chroma_collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    return chroma_collection

#TODO: write assistant_conversation_without_chroma() and test 1 vs the other

def assistant_conversation(openaiClient=client):
    # embeddings_collection = embed_pdf("RAG_files/etsi_pdf.pdf")
    embeddings_collection = get_chroma_collection()

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

        query_embedding = create_openai_embeddings(user_input)

        # Query using the embedding vector (not the text)
        results = embeddings_collection.query(
            query_embeddings=[query_embedding],  # Changed from query_texts
            n_results=10
        )

        retrieved_documents = results['documents'][0]
        information = "\n\n".join(retrieved_documents)

        response = openaiClient.responses.create(
            model="gpt-4o-mini-2024-07-18",
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
    # chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
    # chroma_client.reset()
    assistant_conversation()