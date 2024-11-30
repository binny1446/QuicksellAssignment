import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from typing import List
from langchain.schema import Document
import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, chunk_overlap=800)
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

 # Correct LLM import for Google Generative AI

def get_conversational_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    prompt_template = """
    Use the following conversation history and context to answer the question. If the answer is not in the context and history, 
    just say "Sorry, I didn't understand your question. Do you want to connect with live agent?". Make sure to provide detailed answers when possible.

    Chat History: {chat_history}
    Context: {context}
    Question: {question}

    Answer:
    """

    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            'prompt': PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
        }
    )
    
    return qa_chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
    ]
    if 'chat_history' in st.session_state:
        del st.session_state['chat_history']

def user_input(user_question):
    chain = get_conversational_chain()

    response = chain({"question": user_question})
    return response['answer']

def multi(chunks):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    chain = ({"doc": lambda x: x.page_content} | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}") | llm | StrOutputParser())

    summaries = chain.batch(chunks, {"max_concurrency": 5})
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"
    
    # The retriever
    retriever = MultiVectorRetriever(vectorstore=vectorstore,  byte_store=store,  id_key=id_key)
    doc_ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Docs linked to summaries
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]})  for i, s in enumerate(summaries)]
    
    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, chunks)))
    

def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot with Memory",
        page_icon="🤖"
    )

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    placeholder = st.empty()
                    placeholder.markdown(response)
            
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
