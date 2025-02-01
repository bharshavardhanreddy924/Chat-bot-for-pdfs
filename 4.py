import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import PyPDF2
import torch
from concurrent.futures import ThreadPoolExecutor
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Download NLTK data
nltk.download('punkt', quiet=True)

# Download NLT# Pinecone setup
PINECONE_API_KEY = "api_key"
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "document-qa"
# Model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "facebook/bart-large-cnn"

@st.cache_resource
def initialize_pinecone():
    try:
        return Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    generator = pipeline('text2text-generation', model=GENERATOR_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
    return embedding_model, generator

def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_into_chunks(text, chunk_size=1000):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_pinecone_index(pc):
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"Failed to create Pinecone index: {str(e)}")
        return None

def store_in_pinecone(pc, chunks, embeddings):
    index = create_pinecone_index(pc)
    if index is None:
        return
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        i_end = min(i+batch_size, len(chunks))
        ids = [str(j) for j in range(i, i_end)]
        metadatas = [{"text": chunk} for chunk in chunks[i:i_end]]
        embeddings_batch = embeddings[i:i_end].tolist()
        to_upsert = list(zip(ids, embeddings_batch, metadatas))
        try:
            index.upsert(vectors=to_upsert)
        except Exception as e:
            st.error(f"Failed to upsert vectors: {str(e)}")

def retrieve_chunks(pc, query, embedding_model, top_k=3):
    try:
        index = pc.Index(INDEX_NAME)
        query_embedding = embedding_model.encode([query])[0].tolist()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return [result['metadata']['text'] for result in results['matches']]
    except Exception as e:
        st.error(f"Failed to retrieve chunks: {str(e)}")
        return []

def generate_answer(query, context, generator):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = generator(prompt, max_length=150, num_return_sequences=1, do_sample=False)
        return response[0]['generated_text']
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "I'm sorry, I couldn't generate an answer. Please try rephrasing your question."

def qa_bot(pc, query, embedding_model, generator):
    try:
        relevant_chunks = retrieve_chunks(pc, query, embedding_model)
        context = " ".join(relevant_chunks)
        answer = generate_answer(query, context, generator)
        return answer, relevant_chunks
    except Exception as e:
        st.error(f"Error in QA bot: {str(e)}")
        return "I'm sorry, an error occurred. Please try again.", []

def process_query(pc, query, embedding_model, generator):
    start_time = time.time()
    answer, relevant_chunks = qa_bot(pc, query, embedding_model, generator)
    end_time = time.time()
    processing_time = end_time - start_time
    return answer, relevant_chunks, processing_time

def main():
    st.title("Document QA Bot")

    pc = initialize_pinecone()
    if pc is None:
        st.error("Failed to initialize Pinecone. Please check your API key and try again.")
        return

    embedding_model, generator = load_models()

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            text = process_pdf(uploaded_file)
            chunks = split_into_chunks(text)
            embeddings = embedding_model.encode(chunks)
            store_in_pinecone(pc, chunks, embeddings)
        st.success("Document processed and stored successfully!")

    st.subheader("Ask questions about the document")
    
    # Create a text input for the query
    query = st.text_input("Enter your question:", key="query_input")

    # Create a button to submit the query
    if st.button("Submit Question"):
        if query:
            with st.spinner("Generating answer..."):
                answer, relevant_chunks, processing_time = process_query(pc, query, embedding_model, generator)
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Processing Time:")
            st.write(f"{processing_time:.2f} seconds")
            
            st.subheader("Relevant Document Segments:")
            for i, chunk in enumerate(relevant_chunks):
                with st.expander(f"Segment {i+1}"):
                    st.write(chunk)

    # Multiple queries handling
    st.subheader("Multiple Queries")
    num_queries = st.number_input("Number of queries to process:", min_value=1, max_value=10, value=3)
    queries = [st.text_input(f"Query {i+1}:") for i in range(num_queries)]
    
    if st.button("Process Multiple Queries"):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda q: process_query(pc, q, embedding_model, generator), queries))
        
        for i, (query, (answer, relevant_chunks, processing_time)) in enumerate(zip(queries, results)):
            st.subheader(f"Query {i+1}: {query}")
            st.write("Answer:", answer)
            st.write(f"Processing Time: {processing_time:.2f} seconds")
            with st.expander("Relevant Document Segments"):
                for j, chunk in enumerate(relevant_chunks):
                    st.write(f"Segment {j+1}: {chunk}")

if __name__ == "__main__":
    main()
    
