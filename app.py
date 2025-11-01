import os
import json
import re
from urllib.request import urlopen
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# ---------------------- Load API Key ----------------------
load_dotenv()
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------- Helper Functions ----------------------
def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id: str) -> str:
    """Fetch YouTube transcript as plain text."""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in fetched_transcript)
        return transcript
    except Exception as e:
        st.error(f"Transcript Error: {e}")
        return None

def get_video_metadata(video_id: str):
    """Fetch YouTube video title and thumbnail."""
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = json.loads(urlopen(url).read())
        return response.get("title"), response.get("thumbnail_url")
    except:
        return None, None

# ---------------------- Streamlit UI Config ----------------------
st.set_page_config(
    page_title="🎥 YouTube AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#00B4D8;'>🎬 YouTube AI Assistant</h1>
    <p style='text-align:center; color:gray;'>Ask intelligent questions or get smart summaries from any YouTube video using AI.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Sidebar setup
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.info("Enter your own API key or use the default one stored securely.")
    user_api_key = st.text_input("🔑 Your Google Gemini API key (optional)", type="password")
    temperature = st.slider("🔥 Creativity (Temperature)", 0.0, 1.0, 0.4)
    top_k = st.slider("📄 Retrieved Chunks (k)", 1, 10, 4)
    st.markdown("---")
    st.caption("Developed by **Pankaj Kumawat** 💻")

# ---------------------- Main Input ----------------------
video_url = st.text_input("📺 Paste YouTube Video Link:")
question = st.text_area("💭 Ask a question (or leave blank to summarize):")

col1, col2 = st.columns(2)
ask_btn = col1.button("🔍 Get Answer")
summary_btn = col2.button("🧠 Summarize Video")

# ---------------------- Session State ----------------------
if "video_cache" not in st.session_state:
    st.session_state.video_cache = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------- Main Logic ----------------------
if ask_btn or summary_btn:
    api_key = user_api_key if user_api_key else DEFAULT_API_KEY
    if not api_key:
        st.error("⚠️ No Gemini API key found. Please add your key or set it in the .env file.")
        st.stop()

    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube link. Please enter a valid video URL.")
        st.stop()

    with st.spinner("⏳ Processing your video..."):
        try:
            # Cache check
            if video_id in st.session_state.video_cache:
                data = st.session_state.video_cache[video_id]
                vector_store, transcript = data["vector_store"], data["transcript"]
                title, thumbnail = data["title"], data["thumbnail"]
                st.info("✅ Using cached video data.")
            else:
                transcript = get_transcript(video_id)
                if not transcript:
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embedding=embeddings)
                title, thumbnail = get_video_metadata(video_id)

                st.session_state.video_cache[video_id] = {
                    "vector_store": vector_store,
                    "transcript": transcript,
                    "title": title,
                    "thumbnail": thumbnail,
                }

            if title:
                st.subheader(f"🎞️ {title}")
            if thumbnail:
                st.image(thumbnail, use_container_width=True)

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            context_text = transcript if summary_btn else "\n\n".join(
                doc.page_content for doc in retriever.invoke(question)
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=temperature,
                google_api_key=api_key
            )

            if summary_btn:
                prompt_text = """
                Summarize the following YouTube transcript clearly and concisely.
                Include the main ideas, key insights, and tone of the video.

                Transcript:
                {context}
                """
            else:
                prompt_text = """
                You are a helpful AI assistant. Use the transcript context only to answer the question.
                If the answer is not found, say: "I couldn't find that information in the video."

                Context:
                {context}

                Question: {question}
                """

            prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
            final_prompt = prompt.format(context=context_text, question=question)
            answer = llm.invoke(final_prompt)
            result = answer.content if hasattr(answer, "content") else str(answer)

            if not summary_btn:
                st.session_state.chat_history.append((question, result))

            st.success("✅ Done!")
            st.subheader("💬 AI Response")
            st.write(result)

            st.download_button("⬇️ Download Answer", result, file_name="youtube_ai_answer.txt")

            with st.expander("📝 View Full Transcript"):
                st.text_area("Full Transcript", transcript, height=300)

            if st.session_state.chat_history:
                st.subheader("💬 Chat History")
                for q, a in st.session_state.chat_history[-5:]:
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {a}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error: {e}")






