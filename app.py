from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.request import urlopen
import json
import streamlit as st

# ---------------------- Helper Function ----------------------

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

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="YouTube AI Assistant", page_icon="🎥", layout="wide")
st.title("🎬 YouTube AI Assistant")
st.write("Ask intelligent questions or get summaries from any YouTube video using AI!")

# Sidebar setup
with st.sidebar:
    st.header("⚙️ Settings")
    google_api_key = st.text_input("Enter your Google Gemini API key", type="password")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3)
    top_k = st.slider("Number of Retrieved Chunks (k)", 1, 10, 4)

# Main inputs
video_id = st.text_input("📺 Enter YouTube Video ID (example: Gfr50f6ZBvo):")
question = st.text_area("❓ Ask a question about the video (or leave blank to summarize):")

# ---------------------- Session State Cache ----------------------

if "video_cache" not in st.session_state:
    st.session_state.video_cache = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store tuples of (user_question, ai_answer)

# ---------------------- Display Video Metadata ----------------------

if video_id:
    title, thumbnail = get_video_metadata(video_id)
    if title:
        st.subheader(f"🎞️ {title}")
    if thumbnail:
        st.image(thumbnail, use_container_width=True)

# ---------------------- Main Logic ----------------------

col1, col2 = st.columns(2)
ask_btn = col1.button("🔍 Get Answer")
summary_btn = col2.button("🧠 Summarize Video")

if ask_btn or summary_btn:
    if not google_api_key:
        st.error("Please enter your Google Gemini API key in the sidebar.")
    elif not video_id:
        st.error("Please enter a valid Video ID.")
    else:
        with st.spinner("⏳ Processing..."):
            try:
                # Cache check
                if video_id in st.session_state.video_cache:
                    vector_store = st.session_state.video_cache[video_id]["vector_store"]
                    transcript = st.session_state.video_cache[video_id]["transcript"]
                    st.info("✅ Using cached data for this video.")
                else:
                    transcript = get_transcript(video_id)
                    if not transcript:
                        st.stop()

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.create_documents([transcript])

                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

                    st.session_state.video_cache[video_id] = {
                        "vector_store": vector_store,
                        "transcript": transcript,
                    }

                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
                context_text = transcript if summary_btn else "\n\n".join(
                    doc.page_content for doc in retriever.invoke(question)
                )

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=temperature,
                    google_api_key=google_api_key
                )

                # Prompts
                if summary_btn:
                    prompt_text = """
                    Summarize the following YouTube transcript clearly and concisely.
                    Mention key points, main ideas, and tone.

                    Transcript:
                    {context}
                    """
                else:
                    prompt_text = """
                    You are a helpful assistant. Use only the transcript context to answer.
                    If the answer isn't found, say: "I couldn't find that information in the video."

                    Context:
                    {context}

                    Question: {question}
                    """

                prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
                final_prompt = prompt.format(context=context_text, question=question)

                answer = llm.invoke(final_prompt)

                result = answer.content if hasattr(answer, "content") else str(answer)

                # Save chat history
                if not summary_btn:
                    st.session_state.chat_history.append((question, result))

                # Display
                st.success("✅ Done!")
                st.subheader("💬 Answer:")
                st.write(result)

                # Download option
                st.download_button("⬇️ Download Answer", result, file_name="youtube_ai_answer.txt")

                # Transcript view
                with st.expander("📝 View Full Transcript"):
                    st.text_area("Full Transcript", transcript, height=300)

                # Chat history
                if st.session_state.chat_history:
                    st.subheader("💬 Chat History")
                    for q, a in st.session_state.chat_history[-5:]:
                        st.markdown(f"**Q:** {q}")
                        st.markdown(f"**A:** {a}")
                        st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error: {e}")






















# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from youtube_transcript_api import YouTubeTranscriptApi
# import streamlit as st

# # ---------------------- Helper Function ----------------------

# def get_transcript(video_id: str) -> str:
#     """Fetch YouTube transcript as plain text."""
#     try:
#         ytt_api = YouTubeTranscriptApi()
#         fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
#         transcript = " ".join(chunk.text for chunk in fetched_transcript)
#         return transcript
#     except Exception as e:
#         st.error(f"Transcript Error: {e}")
#         return None

# # ---------------------- Streamlit UI ----------------------

# st.set_page_config(page_title="YouTube AI Assistant", page_icon="🎥", layout="wide")
# st.title("🎬 YouTube AI Assistant")
# st.write("Ask questions about any YouTube video instantly using AI!")

# # Sidebar setup
# with st.sidebar:
#     st.header("⚙️ Settings")
#     google_api_key = st.text_input("Enter your Google Gemini API key", type="password")

# # Main inputs
# video_id = st.text_input("📺 Enter YouTube Video ID:")
# question = st.text_area("❓ Enter your question about the video:")

# # ---------------------- Session State Cache ----------------------

# if "video_cache" not in st.session_state:
#     st.session_state.video_cache = {}  # {video_id: {"vector_store": FAISS, "transcript": str}}

# # ---------------------- Main Logic ----------------------

# if st.button("🔍 Get Answer"):
#     if not google_api_key:
#         st.error("Please enter your Google Gemini API key in the sidebar.")
#     elif not video_id or not question:
#         st.error("Please enter both Video ID and Question.")
#     else:
#         with st.spinner("Processing..."):
#             try:
#                 # 🔹 Check cache first
#                 if video_id in st.session_state.video_cache:
#                     st.info("Using cached data for this video.")
#                     vector_store = st.session_state.video_cache[video_id]["vector_store"]
#                 else:
#                     st.info("Fetching and processing transcript...")

#                     # 1️⃣ Get transcript
#                     transcript = get_transcript(video_id)
#                     if not transcript:
#                         st.stop()

#                     # 2️⃣ Split transcript into chunks
#                     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                     chunks = splitter.create_documents([transcript])

#                     # 3️⃣ Create embeddings & FAISS index
#                     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#                     vector_store = FAISS.from_documents(chunks, embedding=embeddings)

#                     # 🔹 Save in cache
#                     st.session_state.video_cache[video_id] = {
#                         "vector_store": vector_store,
#                         "transcript": transcript,
#                     }

#                 # 4️⃣ Retrieve relevant chunks
#                 retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#                 retrieved_docs = retriever.invoke(question)
#                 context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

#                 # 5️⃣ LLM setup (Gemini)
#                 llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=google_api_key)

#                 # 6️⃣ Prompt template
#                 prompt = PromptTemplate(
#                     template="""
#                     You are a helpful assistant.
#                     Use only the given transcript context to answer.
#                     If the answer is not in the context, reply: "I couldn't find that information in the video."

#                     Context:
#                     {context}

#                     Question: {question}
#                     """,
#                     input_variables=["context", "question"]
#                 )

#                 final_prompt = prompt.format(context=context_text, question=question)

#                 # 7️⃣ Get answer
#                 answer = llm.invoke(final_prompt)

#                 # 8️⃣ Display
#                 st.success("✅ Answer generated successfully!")
#                 st.subheader("💬 Answer:")
#                 st.write(answer.content if hasattr(answer, "content") else answer)

#             except Exception as e:
#                 st.error(f"❌ Error: {e}")
