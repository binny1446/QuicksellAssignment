#writefile app.py
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from sentence_transformers import SentenceTransformer
import faiss
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Configure Gemini API
genai.configure(api_key="AIzaSyC2nRjklHRNgND98qIvCV7KNsmHLBI9yA4")
llm2 = genai.GenerativeModel('gemini-pro')
llm = GoogleGenerativeAI(model="gemini-pro")

# Step 1: Load and encode the FAQ data using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example FAQs (replace with your own set)
faqs = [
    {"question": "What is the return policy?", "answer": "To return a product, log in to your account, select 'Orders', and click on 'Return'."},
    {"question": "How do I track my order?", "answer": "To track your order, go to 'My Orders' and select 'Track Order'."},
    # Add more FAQs here...
]

# Encode FAQ questions
faq_questions = [faq["question"] for faq in faqs]
faq_embeddings = model.encode(faq_questions)

# Build FAISS index for FAQ retrieval
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

# Function to retrieve relevant FAQ based on a query
def retrieve_faq(query, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return faqs[indices[0][0]]['answer']  # Return the answer of the retrieved FAQ

# Step 2: Define tools for intent classification, NER, and FAQ handling
# Intent Classification Tool
intent_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Classify the intent of the following sentence into one of these categories: "
        "1. Product search, 2. FAQ inquiry, 3. Order tracking, 4. General chat. "
        "The sentence is: {user_input}"
    ),
)
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

def classify_intent(user_input):
    response = intent_chain.run(user_input)
    # Extract the intent from the response (e.g., "Product search")
    if "Product search" in response:
        return "Product search"
    elif "FAQ inquiry" in response:
        return "FAQ inquiry"
    elif "Order tracking" in response:
        return "Order tracking"
    elif "General chat" in response:
        return "General chat"
    else:
        return "General chat"  # Default to general chat

# Named Entity Recognition (NER) Tool
ner_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Perform named entity recognition on the following e-commerce sentence: {user_input}. "
        "Extract entities like product names, locations, and stores."
    ),
)
ner_chain = LLMChain(llm=llm, prompt=ner_prompt)

def perform_ner(user_input):
    return ner_chain.run(user_input)

# FAQ Inquiry Tool (RAG tool)
faq_prompt = PromptTemplate(
    input_variables=["retrieved_faq", "user_input"],
    template=(
        "Here is an FAQ relevant to the user's query: {retrieved_faq}. "
        "Explain this FAQ in more detail to the customer based on their input: {user_input}."
    ),
)
faq_chain = LLMChain(llm=llm, prompt=faq_prompt)

def handle_faq(user_input, retrieved_faq):
    return faq_chain.run({"user_input": user_input, "retrieved_faq": retrieved_faq})

# Define the state
class State(TypedDict):
    user_input: str
    intent: str
    ner_result: str
    retrieved_faq: str
    response: str
    messages: Annotated[list, add_messages]

# Add nodes
def intent_classification(state: State) -> State:
    intent = classify_intent(state["messages"][-1])
    return {"intent": intent}

def ner_product_search(state: State) -> State:
    ner_result = perform_ner(state["messages"][-1])
    return {"ner_result": ner_result}

def llm_product_search(state: State) -> State:
    response = llm2.generate_content(f"Search products related to: {state['ner_result'][-1]}").text
    return {"response": response}

def general_chat(state: State) -> State:
    string = " "
    for i in range(len(state["messages"])-1):
        string += str(state["messages"][i])

    string += "Only this part is the current question being asked before other than this statement everything is memory hence only answer this question:"
    string += str(state["messages"][-1])
    response = llm2.generate_content(string).text

    return {"response": response}

def faq_inquiry(state: State) -> State:
    retrieved_faq = retrieve_faq(state["messages"][-1])
    return {"retrieved_faq": retrieved_faq}

def faq_answer(state: State) -> State:
    response = handle_faq(state["messages"][-1], state["retrieved_faq"][-1])
    return {"response": response}

def ner_order_tracking(state: State) -> State:
    ner_result = perform_ner(state["messages"][-1])
    return {"ner_result": ner_result}

def llm_order_tracking(state: State) -> State:
    response = llm2.generate_content(f"Track order with the following details: {state['ner_result'][-1]}").text
    return {"response": response}

def end(state: State) -> State:
    return {"response": "Goodbye!"}

# Initialize the graph
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("INTENT_CLASSIFICATION", intent_classification)
graph.add_node("NER_PRODUCT_SEARCH", ner_product_search)
graph.add_node("LLM_PRODUCT_SEARCH", llm_product_search)
graph.add_node("GENERAL_CHAT", general_chat)
graph.add_node("FAQ_INQUIRY", faq_inquiry)
graph.add_node("FAQ_ANSWER", faq_answer)
graph.add_node("NER_ORDER_TRACKING", ner_order_tracking)
graph.add_node("LLM_ORDER_TRACKING", llm_order_tracking)
graph.add_node("END", end)

# Add edges
graph.add_conditional_edges(
    source="INTENT_CLASSIFICATION",
    path=lambda state: state["intent"],
    path_map={
        "Product search": "NER_PRODUCT_SEARCH",
        "General chat": "GENERAL_CHAT",
        "FAQ inquiry": "FAQ_INQUIRY",
        "Order tracking": "NER_ORDER_TRACKING"
    }
)

graph.add_edge("NER_PRODUCT_SEARCH", "LLM_PRODUCT_SEARCH")
graph.add_edge("LLM_PRODUCT_SEARCH", "END")

graph.add_edge("GENERAL_CHAT", "END")

graph.add_edge("FAQ_INQUIRY", "FAQ_ANSWER")
graph.add_edge("FAQ_ANSWER", "END")

graph.add_edge("NER_ORDER_TRACKING", "LLM_ORDER_TRACKING")
graph.add_edge("LLM_ORDER_TRACKING", "END")

# Set the entry point
graph.set_entry_point("INTENT_CLASSIFICATION")

# Compile the graph
graph = graph.compile(checkpointer=memory)

# Step 4: Define the chatbot agent with memory and continuous interaction
def chatbot_agent():
    config = {"configurable": {"thread_id": "1"}}
    st.session_state.messages = st.session_state.get("messages", [])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("You: "):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if prompt.lower() in ["exit", "quit"]:
            st.session_state.messages.append({"role": "assistant", "content": "Goodbye!"})
            with st.chat_message("assistant"):
                st.markdown("Goodbye!")
            return

        # The config is the **second positional argument** to stream() or invoke()!
        events = graph.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                response = event["messages"][-1]
            if "response" in event:
                response = event["response"]

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Streamlit UI
st.title("Chatbot Interface")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Run the chatbot agent
chatbot_agent()
