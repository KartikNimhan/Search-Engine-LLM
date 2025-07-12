import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---- Tool Setup ----
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# ---- Streamlit App UI ----
st.set_page_config(page_title="LangChain Web Search Chatbot")
st.title("🔎 LangChain - Chat with Search")

st.markdown(
    "Ask anything! This chatbot can search the web using **DuckDuckGo**, **Wikipedia**, and **Arxiv** via LangChain tools."
)

# ---- Sidebar ----
st.sidebar.title("🔐 Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Ensure API key is provided
if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# ---- Chat History Setup ----
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---- Handle User Input ----
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Create chat prompt history
    full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"]])

    # Setup LangChain agent
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            # Smart routing: direct arXiv ID queries to arxiv tool
            if "arxiv:" in prompt.lower():
                response = arxiv.run(prompt)
            else:
                response = search_agent.run(full_prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"❌ Error: {str(e)}"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)
