import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
from langchain.schema import Document

# Streamlit setup
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 Langchain: Summarize Text from YT or Website")
st.subheader("Provide a YouTube or Website URL")
st.secrets["HF_TOKEN"]

# Sidebar for Groq API
with st.sidebar:
    groq_api_key = st.text_input("🔑 Groq API Key", type="password")

# Input URL
generic_url = st.text_input("Paste a YouTube or Website URL")

# Prompt template
prompt_template = """
Provide a concise and informative summary of the following content in under 300 words:

Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Load LLM
if groq_api_key:
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
else:
    st.warning("Please enter your Groq API key to proceed.")
    st.stop()


def get_youtube_transcript(video_url):
    """Extract transcript from YouTube video"""
    video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", video_url)
    if not video_id_match:
        return None
    video_id = video_id_match.group(1)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([entry["text"] for entry in transcript])
    return full_text


# Button click
if st.button("Summarize the content from YT or Website"):

    if not generic_url.strip():
        st.error("❌ Please enter a valid URL.")
    elif not validators.url(generic_url):
        st.error("❌ URL is not valid.")
    else:
        try:
            with st.spinner("⏳ Processing..."):

                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        transcript_text = get_youtube_transcript(generic_url)
                        if not transcript_text:
                            st.error("❌ Could not extract transcript.")
                            st.stop()
                        docs = [Document(page_content=transcript_text)]
                    except (TranscriptsDisabled, NoTranscriptFound):
                        st.error("❌ No transcript available for this YouTube video.")
                        st.stop()
                else:
                    # Website URL loader
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/115.0.0.0 Safari/537.36"
                            )
                        }
                    )
                    docs = loader.load()

                if not docs:
                    st.warning("⚠️ No content loaded. Try another URL.")
                    st.stop()

                # Summarize
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success("✅ Summary Generated")
                st.write(summary)

        except Exception as e:
            st.error("❌ Exception occurred while summarizing the content.")
            st.exception(e)
