from chromadb import Settings
from openai import OpenAI
import chromadb
import os
import re
from dotenv import load_dotenv

from helper_utils import word_wrap
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
# VECTOR_STORE_ID="vs_67ea76dc84988191b848a57b0f6b7b8e"

client = OpenAI(api_key=OPENAI_API_KEY)

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
    return chroma_client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"}
    )


def extract_section_number(heading_text):
    """Extract section number from a heading text."""
    match = re.match(r'^(\d+(?:\.\d+)*)', heading_text.strip())
    if match:
        return match.group(1)
    return ""


def count_section_levels(section_number):
    """Count the number of levels in a section number."""
    if not section_number:
        return 0
    return len(section_number.split('.'))


def identify_headings(text):
    """Identify headings in the document based on patterns."""
    # Patterns for section headings in ETSI technical specs
    heading_patterns = [
        r'^\d+\.\d+\.\d+\.\d+\s+.*$',  # 1.2.3.4 Heading text
        r'^\d+\.\d+\.\d+\s+.*$',  # 1.2.3 Heading text
        r'^\d+\.\d+\s+.*$',  # 1.2 Heading text
        r'^\d+\s+.*$',  # 1 Heading text
    ]

    lines = text.split('\n')
    heading_indices = []
    heading_levels = []

    for i, line in enumerate(lines):
        line = line.strip()
        for pattern in heading_patterns:
            if re.match(pattern, line):
                heading_indices.append(i)
                section_num = extract_section_number(line)
                level = count_section_levels(section_num)
                heading_levels.append(level)
                break

    return heading_indices, heading_levels, lines


def create_hierarchical_chunks(text):
    heading_indices, heading_levels, lines = identify_headings(text)

    if not heading_indices:
        return [text], []

    # Add document start if not included
    if 0 not in heading_indices:
        heading_indices.insert(0, 0)
        heading_levels.insert(0, 0)  # Root level

    # Add document end for the last chunk
    heading_indices.append(len(lines))
    heading_levels.append(0)

    # Sort indices for proper sectioning
    sorted_indices = sorted(range(len(heading_indices)), key=lambda k: heading_indices[k])
    heading_indices = [heading_indices[i] for i in sorted_indices]
    heading_levels = [heading_levels[i] for i in sorted_indices]

    chunks = []
    chunk_metadata = []

    # Debug information
    print(f"Total headings found: {len(heading_indices) - 2}")  # -2 for start/end markers
    for i in range(len(heading_indices) - 1):
        if i > 0 and i < len(heading_indices) - 1:  # Skip start/end markers
            print(f"Heading {i}: {lines[heading_indices[i]].strip()} (Level {heading_levels[i]})")

    # Process each heading to create chunks with proper content
    for i in range(len(heading_indices) - 1):
        start = heading_indices[i]
        curr_level = heading_levels[i]

        # Skip artificial start marker
        if i == 0 and curr_level == 0:
            continue

        heading = lines[start].strip() if start < len(lines) else ""
        section_num = extract_section_number(heading)

        # Find the end of this section (next heading of same or higher level)
        end = None
        for j in range(i + 1, len(heading_indices)):
            if heading_levels[j] <= curr_level:
                end = heading_indices[j]
                break

        # If no natural end found, use the next heading
        if end is None:
            end = heading_indices[i + 1]

        # Extract content for this section (excluding subsections)
        direct_content_end = heading_indices[i + 1] if i + 1 < len(heading_indices) else len(lines)
        for j in range(i + 1, len(heading_indices) - 1):
            if heading_levels[j] > curr_level:  # Found a subsection
                direct_content_end = heading_indices[j]
                break

        # Get direct content of this section (excluding subsection content)
        direct_content = '\n'.join(lines[start:direct_content_end]).strip()

        # Get full content including subsections
        full_content = '\n'.join(lines[start:end]).strip()

        # Skip truly empty sections
        if not full_content:
            continue

        # Get parent section info for context
        parent_section = None
        parent_heading = None

        if curr_level > 1:  # Has a parent section
            # Look backward to find the parent section
            for j in range(i - 1, -1, -1):
                parent_level = heading_levels[j]
                if parent_level > 0 and parent_level < curr_level:
                    parent_heading = lines[heading_indices[j]].strip()
                    parent_section = extract_section_number(parent_heading)
                    break

        # Build metadata based on section level
        if curr_level == 3:  # Detail level (e.g., 5.3.2)
            metadata = {
                "section": section_num,
                "heading": heading,
                "level": curr_level,
                "parent_section": parent_section,
                "parent_heading": parent_heading,
                "chunk_type": "detail"
            }
            chunks.append(full_content)
            chunk_metadata.append(metadata)

        elif curr_level == 2:  # Overview level (e.g., 5.3)
            # Find all subsections
            subsections = []
            for j in range(i + 1, len(heading_indices) - 1):
                if heading_levels[j] <= curr_level:  # We've moved past subsections
                    break
                if heading_levels[j] == curr_level + 1:  # Direct subsection
                    subsection = lines[heading_indices[j]].strip()
                    subsections.append(subsection)

            subsection_info = "\n".join(subsections) if subsections else "No subsections"

            # Don't create a chunk if there's only a heading and no real content
            if len(direct_content.split('\n')) > 1 or not all(
                    subsection in direct_content for subsection in subsections):
                metadata = {
                    "section": section_num,
                    "heading": heading,
                    "level": curr_level,
                    "subsections": subsection_info,
                    "chunk_type": "overview"
                }
                # Use direct_content for level 2 to avoid duplicating subsection content
                chunks.append(direct_content if len(direct_content) > len(heading) else full_content)
                chunk_metadata.append(metadata)

        elif curr_level == 1:  # Major section (e.g., 5)
            metadata = {
                "section": section_num,
                "heading": heading,
                "level": curr_level,
                "chunk_type": "major_section"
            }
            # Use direct_content for level 1 to avoid duplicating subsection content
            chunks.append(direct_content if len(direct_content) > len(heading) else full_content)
            chunk_metadata.append(metadata)

    # Debug output of chunking results
    print(f"\nCreated {len(chunks)} chunks:")
    for i, (chunk, meta) in enumerate(zip(chunks, chunk_metadata)):
        content_preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        print(f"Chunk {i + 1}: Section {meta.get('section', 'N/A')}, Type: {meta.get('chunk_type', 'N/A')}")
        print(f"Content preview: {content_preview}")
        print(f"Content length: {len(chunk)} characters")
        print("-" * 40)

    return chunks, chunk_metadata


def embed_pdf(file_path="RAG_files/etsi_pdf.pdf"):
    chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
    chroma_client.reset()
    chroma_collection = get_chroma_collection()

    if chroma_collection.count() == 0:
        reader = PdfReader(file_path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]

        # Join the PDF texts with page boundaries
        full_text = '\n\n'.join(pdf_texts)

        # First split by large sections to avoid memory issues with very large documents
        large_section_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n"],
            chunk_size=10000,
            chunk_overlap=200
        )
        large_sections = large_section_splitter.split_text(full_text)

        all_chunks = []
        all_metadata = []

        # Process each large section to find headings and create hierarchical chunks
        for section_idx, large_section in enumerate(large_sections):
            section_chunks, section_metadata = create_hierarchical_chunks(large_section)

            # Find page numbers for each chunk
            for i, (chunk, metadata) in enumerate(zip(section_chunks, section_metadata)):
                page_numbers = []
                for p, page_text in enumerate(pdf_texts):
                    # Check if any significant portion of the chunk exists on this page
                    # We look for sentences to avoid false matches with common phrases
                    sentences = [s.strip() for s in chunk.split('. ') if len(s.strip()) > 50]
                    for sentence in sentences:
                        if sentence in page_text:
                            page_numbers.append(p + 1)  # 1-indexed pages
                            break

                # Add page info to metadata and prepare for Chroma
                metadata["page_numbers"] = ','.join(map(str, set(page_numbers))) if page_numbers else ""

                all_chunks.append(chunk)
                all_metadata.append(metadata)

                # Print every chunk with its metadata
                print("\n" + "=" * 80)
                print(f"PRINTING ALL {len(all_chunks)} CHUNKS")
                print("=" * 80 + "\n")

                for i, (chunk, metadata) in enumerate(zip(all_chunks, all_metadata)):
                    if i > 300:
                        break
                    print(f"\nCHUNK #{i + 1}:")
                    print(f"Section: {metadata.get('section', 'N/A')}")
                    print(f"Heading: {metadata.get('heading', 'N/A')}")
                    print(f"Level: {metadata.get('level', 'N/A')}")
                    print(f"Type: {metadata.get('chunk_type', 'N/A')}")
                    print(f"Page Numbers: {metadata.get('page_numbers', 'N/A')}")
                    if 'parent_section' in metadata:
                        print(f"Parent Section: {metadata.get('parent_section', 'N/A')}")
                    if 'parent_heading' in metadata:
                        print(f"Parent Heading: {metadata.get('parent_heading', 'N/A')}")
                    if 'subsections' in metadata:
                        print(f"Subsections: {metadata.get('subsections', 'N/A')}")
                    print("-" * 40)
                    print(f"Content (whole chunk): {chunk}...")
                    print("-" * 80)

        # Create IDs for all chunks
        ids = [str(i) for i in range(len(all_chunks))]

        # Convert metadata to format expected by Chroma
        chroma_metadata = []
        for meta in all_metadata:
            # Convert all metadata values to strings for Chroma
            chroma_meta = {k: str(v) for k, v in meta.items()}
            chroma_metadata.append(chroma_meta)

        # Add documents with their metadata
        chroma_collection.add(
            ids=ids,
            documents=all_chunks,
            metadatas=chroma_metadata
        )

        print(f"Added {chroma_collection.count()} chunks to the collection")

        # Print some example chunks to verify the chunking works as expected
        if chroma_collection.count() > 0:
            examples = chroma_collection.get(ids=ids[:3])
            print("\nExample chunks:")
            for i, (doc, meta) in enumerate(zip(examples["documents"], examples["metadatas"])):
                print(f"\nChunk {i + 1}:")
                print(f"Section: {meta.get('section', 'N/A')}")
                print(f"Heading: {meta.get('heading', 'N/A')}")
                print(f"Level: {meta.get('level', 'N/A')}")
                print(f"Type: {meta.get('chunk_type', 'N/A')}")
                print(f"First 200 chars: {doc[:200]}...")

    return chroma_collection


def expand_query(query):
    """Generate variations of the query to improve retrieval."""
    expanded_queries = [query]

    # Add a sections-focused version for structural questions
    if "section" in query.lower() or "procedure" in query.lower():
        expanded_queries.append(f"section about {query}")

    # Add a specific telecom-focused version for domain-specific queries
    telecom_terms = ["vlr", "mme", "eps", "sgs", "alert", "procedure", "paging", "detach", "imsi"]
    if any(term in query.lower() for term in telecom_terms):
        expanded_queries.append(f"telecommunications specification for {query}")

    # Add procedural variations for "what are" questions
    if "what are" in query.lower() or "how does" in query.lower():
        procedure_query = query.lower().replace("what are", "steps in").replace("how does", "procedure for")
        expanded_queries.append(procedure_query)

    return expanded_queries


def hybrid_search(collection, query, n_results=30):
    """Perform a hybrid search strategy prioritizing relevant section types."""
    expanded_queries = expand_query(query)

    # First try to find exact section matches if the query looks like it's asking about a specific section
    section_match = re.search(r'(section|procedure|chapter)\s+(\d+\.\d+(?:\.\d+)?)', query, re.IGNORECASE)
    section_results = []

    if section_match:
        section_num = section_match.group(2)
        # Query specifically for that section number
        section_results = collection.query(
            query_texts=[""],  # Empty query to match only on metadata
            n_results=5,
            where={"section": {"$eq": section_num}},
            include=["documents", "metadatas", "distances"]
        )

    # Get results from vector search with expanded queries
    all_results = []
    for exp_query in expanded_queries:
        results = collection.query(
            query_texts=exp_query,
            n_results=n_results // 2,
            include=["documents", "metadatas", "distances"]
        )
        all_results.append(results)

    # Combine and deduplicate results
    combined_docs = []
    combined_metadata = []
    seen_docs = set()

    # First add section-specific results if any
    if section_match and 'documents' in section_results:
        for i, (doc, meta) in enumerate(zip(section_results['documents'][0], section_results['metadatas'][0])):
            doc_hash = hash(doc[:100])
            if doc_hash not in seen_docs:
                combined_docs.append(doc)
                combined_metadata.append(meta)
                seen_docs.add(doc_hash)

    # Then add results from semantic search
    for result in all_results:
        for i, (docs, meta) in enumerate(zip(result['documents'][0], result['metadatas'][0])):
            doc_hash = hash(docs[:100])
            if doc_hash not in seen_docs:
                combined_docs.append(docs)
                combined_metadata.append(meta)
                seen_docs.add(doc_hash)

    # Prioritize chunks with the right level of detail based on the query
    prioritized_docs = []
    prioritized_metadata = []

    # First add detail chunks (level 3) related to the query
    for doc, meta in zip(combined_docs, combined_metadata):
        if meta.get('chunk_type') == 'detail':
            prioritized_docs.append(doc)
            prioritized_metadata.append(meta)

    # Then add overview chunks (level 2) for context
    for doc, meta in zip(combined_docs, combined_metadata):
        if meta.get('chunk_type') == 'overview' and doc not in prioritized_docs:
            prioritized_docs.append(doc)
            prioritized_metadata.append(meta)

    # Finally add any remaining chunks
    for doc, meta in zip(combined_docs, combined_metadata):
        if doc not in prioritized_docs:
            prioritized_docs.append(doc)
            prioritized_metadata.append(meta)

    # Limit to requested number of results
    return {
        "documents": prioritized_docs[:n_results],
        "metadatas": prioritized_metadata[:n_results]
    }


def find_related_section_chunks(collection, section_numbers):
    """Find chunks related to the same sections or parent/child sections."""
    if not section_numbers:
        return [], []

    related_docs = []
    related_metadata = []
    seen_docs = set()

    for section in section_numbers:
        if not section:
            continue

        # Find parent section (e.g., for 5.3.2 get 5.3)
        parts = section.split('.')
        if len(parts) > 2:
            parent_section = '.'.join(parts[:-1])

            # Get the parent section overview
            parent_results = collection.query(
                query_texts=[""],  # Empty query to match only on metadata
                n_results=2,
                where={"section": {"$eq": parent_section}},
                include=["documents", "metadatas"]
            )

            # Add parent chunks
            if 'documents' in parent_results and parent_results['documents']:
                for doc, meta in zip(parent_results['documents'][0], parent_results['metadatas'][0]):
                    doc_hash = hash(doc[:100])
                    if doc_hash not in seen_docs:
                        related_docs.append(doc)
                        related_metadata.append(meta)
                        seen_docs.add(doc_hash)

        # Find sibling sections (e.g., for 5.3.2 get 5.3.1, 5.3.3, etc.)
        if len(parts) >= 3:
            parent_prefix = '.'.join(parts[:-1]) + '.'

            # Get all documents and filter on our side
            all_sections = collection.query(
                query_texts=[""],  # Empty query to match only on metadata
                n_results=100,  # Get a larger number to filter from
                include=["documents", "metadatas"]
            )

            # Filter for sibling sections manually
            if 'documents' in all_sections and all_sections['documents']:
                for doc, meta in zip(all_sections['documents'][0], all_sections['metadatas'][0]):
                    section_val = meta.get('section', '')
                    # Check if this is a sibling section (starts with the same parent prefix)
                    if section_val.startswith(parent_prefix) and section_val != section:
                        doc_hash = hash(doc[:100])
                        if doc_hash not in seen_docs:
                            related_docs.append(doc)
                            related_metadata.append(meta)
                            seen_docs.add(doc_hash)

    return related_docs, related_metadata


def assistant_conversation(openaiClient=client):
    # embeddings_collection = embed_pdf()
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

        # Step 1: Perform initial hybrid search
        results = hybrid_search(embeddings_collection, user_input, n_results=12)
        retrieved_documents = results['documents']
        retrieved_metadata = results['metadatas']

        # Step 2: Check if we need to find related section chunks to ensure completeness
        procedure_keywords = ["procedure", "alert", "steps", "processes"]
        need_related_search = any(kw in user_input.lower() for kw in procedure_keywords)

        if need_related_search:
            print("Finding complete procedure information...")
            # Extract section numbers from retrieved chunks
            section_numbers = [meta.get("section", "") for meta in retrieved_metadata]

            # Find related section chunks
            related_docs, related_metadata = find_related_section_chunks(
                embeddings_collection,
                section_numbers
            )

            # Add the related chunks to our results
            retrieved_documents.extend(related_docs)
            retrieved_metadata.extend(related_metadata)

        # Step 3: Prepare context with section information
        context_documents = []
        for doc, meta in zip(retrieved_documents, retrieved_metadata):
            section = meta.get('section', '')
            heading = meta.get('heading', '')
            chunk_type = meta.get('chunk_type', '')
            page_str = meta.get('page_numbers', '')

            context_info = []
            if section and heading:
                context_info.append(f"Section {section}: {heading}")
            elif heading:
                context_info.append(f"{heading}")

            if page_str:
                pages = page_str.split(',')
                context_info.append(f"Page(s): {', '.join(pages)}")

            # Add chunk type for clarity
            if chunk_type == 'detail':
                context_info.append("Detailed section")
            elif chunk_type == 'overview':
                context_info.append("Section overview")

            context_header = f"[{'; '.join(context_info)}]" if context_info else ""

            if context_header:
                context_documents.append(f"{context_header}\n{doc}")
            else:
                context_documents.append(doc)

        # Step 4: Format the context and send to LLM
        information = "\n\n---\n\n".join(context_documents)

        prompt = f"""
Question: {user_input}

Relevant Information from Technical Document:
{information}

Please provide a comprehensive answer that includes ALL steps, procedures, or details mentioned in the technical document about this topic. 
If there are numbered steps or a specific order to a procedure, make sure to include all of them in your answer.
If the information mentions multiple cases or scenarios (such as normal case, failure case, etc.), include all of them.
"""

        response = openaiClient.responses.create(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.15,
            instructions=SYSTEM_MESSAGE_FOR_RAG,
            input=[
                {
                    "role": "user",
                    "content": prompt,
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