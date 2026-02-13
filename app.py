from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from datetime import datetime

# Load environment variables
load_dotenv()

def create_search_tool(vector_store):
    @tool
    def search_documents(query: str) -> str:
        """
        Searches the company document repository for relevant information based on the given query. Use this to find information about company policies, benefits, and procedures.
        """
        results = vector_store.similarity_search_with_score(query, k=3)
        formatted = []
        for idx, (doc, score) in enumerate(results, 1):
            formatted.append(f"Result {idx} (Score: {score:.4f}): {doc.page_content}")
        return "\n\n".join(formatted)
    return search_documents

def load_with_markdown_structure_chunking(vector_store, file_path):
    """
    Reads a markdown file, splits it by headers, then further chunks by size with overlap, and loads into the vector store.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
        )
        header_chunks = header_splitter.split_text(text)
        # header_chunks is a list of Document objects
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200
        )
        all_chunks = []
        for doc in header_chunks:
            # Split each header chunk further if needed
            sub_chunks = size_splitter.create_documents([doc.page_content])
            # Carry over header metadata to sub-chunks
            for sub_chunk in sub_chunks:
                sub_chunk.metadata = {**doc.metadata, **(sub_chunk.metadata or {})}
            all_chunks.extend(sub_chunks)
        print(f"Markdown structure chunking: {len(all_chunks)} chunks (with overlap)")
        stored = load_document_with_chunks(vector_store, file_path, all_chunks)
        print(f"Stored {stored} markdown-structure chunks in the vector store.")
        return stored
    except Exception as e:
        print(f"Error during markdown structure chunked loading: {e}")
        return 0

def load_with_paragraph_chunking(vector_store, file_path):
    """
    Reads a file, splits it into paragraph-based chunks, and loads them into the vector store.
    Prints statistics about the chunking process.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.create_documents([text])
        num_chunks = len(chunks)
        sizes = [len(chunk.page_content) for chunk in chunks]
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        starts_with_newline = sum(1 for chunk in chunks if chunk.page_content.startswith("\n"))
        print(f"Paragraph chunking: {num_chunks} chunks (min: {min_size}, max: {max_size} chars)")
        print(f"Chunks starting with newline: {starts_with_newline}")
        stored = load_document_with_chunks(vector_store, file_path, chunks)
        print(f"Stored {stored} paragraph-based chunks in the vector store.")
        return stored
    except Exception as e:
        print(f"Error during paragraph chunked loading: {e}")
        return 0

def load_with_fixed_size_chunking(vector_store, file_path):
    """
    Reads a file, splits it into fixed-size chunks, and loads them into the vector store.
    Prints statistics about the chunking process.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separator=" "
        )
        chunks = splitter.create_documents([text])
        num_chunks = len(chunks)
        avg_size = sum(len(chunk.page_content) for chunk in chunks) / num_chunks if num_chunks else 0
        print(f"Splitting '{os.path.basename(file_path)}' into {num_chunks} chunks (avg size: {avg_size:.1f} chars)")
        stored = load_document_with_chunks(vector_store, file_path, chunks)
        print(f"Stored {stored} chunks in the vector store.")
        return stored
    except Exception as e:
        print(f"Error during chunked loading: {e}")
        return 0

def load_document_with_chunks(vector_store, file_path, chunks):
    """
    Stores a list of chunked LangChain Document objects in the vector store with updated metadata.
    Returns the total number of chunks stored.
    """
    try:
        total = len(chunks)
        file_name = os.path.basename(file_path)
        for idx, chunk in enumerate(chunks, 1):
            chunk.metadata = dict(chunk.metadata) if chunk.metadata else {}
            chunk.metadata.update({
                'fileName': f"{file_name} (Chunk {idx}/{total})",
                'createdAt': datetime.now().isoformat(),
                'chunkIndex': idx
            })
            vector_store.add_documents([chunk])
            print(f"Stored chunk {idx}/{total} for '{file_name}' ({len(chunk.page_content)} chars)")
        return total
    except Exception as e:
        print(f"Error loading document chunks: {e}")
        return 0

def load_document(vector_store, file_path):
    """
    Loads a document from file_path, creates a Document with metadata, adds it to the vector store, and returns the document ID.
    Handles file errors gracefully.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        metadata = {
            'fileName': os.path.basename(file_path),
            'createdAt': datetime.now().isoformat()
        }
        doc = Document(page_content=text, metadata=metadata)
        doc_ids = vector_store.add_documents([doc])
        print(f"Loaded '{metadata['fileName']}' ({len(text)} chars) into vector store.")
        return doc_ids[0] if doc_ids else None
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        err_msg = str(e)
        if "maximum context length" in err_msg or "token" in err_msg:
            print("⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"Error loading document: {err_msg}")
        return None

def search_sentences(vector_store, query, k=3):
    """
    Search for the top-k most similar sentences in the vector store to the query.
    Prints the results with rank, similarity score, and sentence text.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    print(f"\nTop {k} results for query: '{query}'")
    for rank, (doc, score) in enumerate(results, 1):
        print(f"{rank}. Score: {score:.4f} | Sentence: {doc.page_content}")

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors
    Cosine similarity = (A · B) / (||A|| * ||B||)
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    
    return dot_product / (norm_a * norm_b)

def main():
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("Please create a .env file with your GitHub token:")
        print("GITHUB_TOKEN=your-github-token-here")
        print("\nGet your token from: https://github.com/settings/tokens")
        print("Or use GitHub Models: https://github.com/marketplace/models")
        return
    # Create OpenAIEmbeddings instance for GitHub Models API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        check_embedding_ctx_length=False
    )


    # Create InMemoryVectorStore instance
    vector_store = InMemoryVectorStore(embeddings)

    # Create ChatOpenAI chat model (GitHub Models API)
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN")
    )


    # Create the search tool
    search_tool = create_search_tool(vector_store)

    # Create the prompt template for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions about company policies, benefits, and procedures. Use the search_documents tool to find relevant information before answering. Always cite which document chunks you used in your answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create the AgentExecutor using initialize_agent with ReAct pattern
    agent_executor = initialize_agent(
        tools=[search_tool],
        llm=chat_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        prompt=prompt
    )

    print("\n=== Loading Documents into Vector Database ===")
    file_path = "HealthInsuranceBrochure.md"
    doc_id = load_document(vector_store, file_path)
    if doc_id:
        print(f"Document '{file_path}' loaded successfully with ID: {doc_id}")

    file_path2 = "EmployeeHandbook.md"
    load_with_markdown_structure_chunking(vector_store, file_path2)

    # Agent-powered chat interface
    chat_history = []
    print("\n=== Company Policy Assistant ===")
    print("Ask any question about company policies, benefits, or procedures. The agent will search the documents and answer, citing sources. Type 'quit' or 'exit' to end.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        response = result["output"]
        print(f"Agent: {response}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()
