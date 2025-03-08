import os
import re
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
from io import BytesIO
from youtube_dl import YoutubeDL
import tempfile
import hashlib

# Configure Google Gemini API
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def clean_text(text):
    """Clean text by removing HTML tags, extra spaces, and empty lines."""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single space
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF pages, ignoring pages with no text."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n--- PAGE BREAK ---\n\n"
    return text.strip()

def split_text(text, min_chunk_size=2000, max_chunk_size=4000):
    """Split text into smaller chunks while preserving context."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if (len(current_chunk) + len(para) > max_chunk_size and 
            len(current_chunk) >= min_chunk_size):
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += '\n\n' + para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def run_gemini_task(prompt, text):
    """Send text to Gemini model and handle responses."""
    try:
        response = model.generate_content(f"{prompt}\n\n{text}")
        return response.text
    except Exception as e:
        st.error(f"Error processing text with Gemini: {e}")
        return ""

def is_valid_youtube_url(url):
    """Check if the YouTube URL is valid using youtube-dl."""
    ydl = YoutubeDL()
    try:
        info = ydl.extract_info(url, download=False)
        return True, info.get('title', 'No title available')
    except:
        return False, None

def get_related_videos(topic, max_results=3):
    """Get related YouTube videos for a given topic."""
    search_query = f"{topic} educational video"
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
        'ignoreerrors': True,
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch{max_results}:{search_query}", download=False)
            if search_results and 'entries' in search_results:
                videos = []
                for entry in search_results.get('entries', []):
                    if entry:
                        videos.append({
                            'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                            'title': entry.get('title', 'No title')
                        })
                return videos
        except Exception as e:
            st.error(f"Error finding related videos: {e}")
    
    return []

# Improved function to prepare text for TTS
def prepare_text_for_tts(text):
    """Clean and prepare text for text-to-speech conversion."""
    # Remove markdown symbols and formatting
    text = re.sub(r'[#*`_~\-–—]', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Replace newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Replace common abbreviations
    text = re.sub(r'\bfig\.\s', 'figure ', text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.\s', 'for example, ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi\.e\.\s', 'that is, ', text, flags=re.IGNORECASE)
    # Remove parentheses and brackets with their content if they're short references
    text = re.sub(r'\s\([^)]{1,4}\)', '', text)
    text = re.sub(r'\s\[[^\]]{1,4}\]', '', text)
    # Keep longer parenthetical content
    text = re.sub(r'\(([^)]{5,})\)', r'\1', text)
    text = re.sub(r'\[([^\]]{5,})\]', r'\1', text)
    return text.strip()

# Generate a cache key for TTS content
def get_tts_cache_key(text, lang, speed):
    """Generate a unique key for caching TTS audio."""
    key_content = f"{text}_{lang}_{speed}"
    return hashlib.md5(key_content.encode()).hexdigest()

# Improved TTS function with caching
def text_to_speech(text, lang='en', slow=False):
    """Convert text to speech with caching to avoid regenerating the same audio."""
    # Create a cache if it doesn't exist
    if "tts_cache" not in st.session_state:
        st.session_state.tts_cache = {}
    
    # Clean and prepare text
    cleaned_text = prepare_text_for_tts(text)
    
    # Create a cache key
    cache_key = get_tts_cache_key(cleaned_text, lang, slow)
    
    # Check if we already have this audio in cache
    if cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key]
    
    # If not in cache, generate new audio
    try:
        tts = gTTS(text=cleaned_text, lang=lang, slow=slow)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Store in cache
        audio_data = audio_bytes.read()
        st.session_state.tts_cache[cache_key] = audio_data
        
        return audio_data
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Split long text for TTS to avoid timeouts
def chunk_text_for_tts(text, max_chars=1000):
    """Split text into smaller chunks for TTS processing."""
    # First clean the text
    text = prepare_text_for_tts(text)
    
    # If text is already small enough, return as is
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:  # Save the completed chunk
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Improved prompts with YouTube video handling
explain_prompt = """
### **Transform Educational Content into Interactive Slides**
Divide the content into concise slides (200-400 words each). Each slide must have:
- A clear title starting with "### Slide X: [Title]"
- Bullet points or short paragraphs
- Examples if applicable
- Include a topic for video recommendations in a separate line in the format "VIDEO_TOPIC: topic keywords" (This line will be removed from the displayed content)

🎯 **Example Format:**
### Slide 1: Introduction to AI
Artificial Intelligence (AI) refers to systems that perform tasks requiring human intelligence. Examples include:
- Speech recognition
- Image classification
- Self-driving cars

VIDEO_TOPIC: artificial intelligence basics

---

### Slide 2: Machine Learning Basics
Machine learning is a subset of AI. Key concepts:
- **Training Data:** Used to teach models
- **Algorithms:** Define how models learn
- **Predictions:** Model outputs based on learned patterns

**Explanation:** Training data is like textbooks for models to learn from.

VIDEO_TOPIC: machine learning fundamentals

---

**Input Text:**
"""

assessment_prompt = """
Generate exactly 10 multiple-choice questions (MCQ) with explanations. Each question must follow this format:
1. [Question]
   A. [Option]
   B. [Option]
   C. [Option]
   D. [Option]
   Correct Answer: [Letter]. Explanation: [Brief context from the slides]

Ensure explanations reference slide content for clarity. Avoid ambiguous answers.

🎯 **Example:**
1. What is supervised learning?
   A. Learning without labels
   B. Learning with labeled data
   C. Learning from rewards
   D. Learning from random patterns
   Correct Answer: B. Explanation: Supervised learning uses labeled data for training (Slide 2).

---

2. What is the role of training data in machine learning?
   A. It is used to test the model
   B. It is used to teach the model
   C. It is used to visualize data
   D. It is used to clean data
   Correct Answer: B. Explanation: Training data is used to teach models (Slide 2).

---

Continue for 10 questions.

**Input Text:**
"""

st.title("📚 AI Tutor Model")

# File uploader
uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

# Initialize session state
if "slides" not in st.session_state:
    st.session_state.slides = []
    st.session_state.titles = []
    st.session_state.current_slide = 0
    st.session_state.questions = []
    st.session_state.correct_answers = []
    st.session_state.video_topics = []
    st.session_state.cached_videos = {}
    st.session_state.tts_cache = {}

if uploaded_file:
    with st.spinner("Extracting text..."):
        lesson_text = extract_text_from_pdf(uploaded_file)
        lesson_text = clean_text(lesson_text)
    
    if st.button("🔍 Generate Slides"):
        try:
            if len(lesson_text) > 1000:
                chunks = split_text(lesson_text)
                presentation = ""
                for chunk in chunks:
                    response = run_gemini_task(explain_prompt, chunk)
                    presentation += f"{response}\n---\n" if response else ""
            else:
                presentation = run_gemini_task(explain_prompt, lesson_text)
            
            if presentation:
                slides = [s.strip() for s in presentation.split("---") if s.strip()]
                titles = []
                video_topics = []
                cleaned_slides = []
                
                # Extract titles and video topics
                for slide in slides:
                    title_match = re.search(r'### Slide \d+: (.+)', slide)
                    title = title_match.group(1).strip() if title_match else "Untitled Slide"
                    titles.append(title)
                    
                    # Extract topic for video recommendation
                    topic_match = re.search(r'VIDEO_TOPIC: (.+?)$', slide, re.MULTILINE)
                    topic = topic_match.group(1).strip() if topic_match else title
                    video_topics.append(topic)
                    
                    # Remove the topic marker from the slide content
                    cleaned_slide = re.sub(r'VIDEO_TOPIC: (.+?)$', '', slide, flags=re.MULTILINE)
                    cleaned_slides.append(cleaned_slide.strip())
                
                st.session_state.slides = cleaned_slides
                st.session_state.titles = titles
                st.session_state.video_topics = video_topics
                st.session_state.cached_videos = {}  # Reset cached videos
                st.session_state.tts_cache = {}  # Reset TTS cache
                st.success("Slides generated successfully!")
            else:
                st.error("No content generated. Check your input file.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.session_state.slides:
        slides = st.session_state.slides
        titles = st.session_state.titles
        video_topics = st.session_state.video_topics
        current_slide = st.session_state.current_slide

        # Sidebar Navigation
        st.sidebar.title("📑 Table of Contents")
        for i, title in enumerate(titles):
            if st.sidebar.button(f"{i+1}. {title}"):
                st.session_state.current_slide = i
                st.rerun()
        
        if st.sidebar.button(f"{len(titles)+1}. Take Assessment"):
            if len(slides) > 0:
                st.session_state.current_slide = len(slides)
                st.rerun()
            else:
                st.warning("No slides available to take assessment.")

        # Content Display
        with st.expander("📖 Slide Content", expanded=True):
            if current_slide < len(slides):
                slide_content = slides[current_slide]
                st.markdown(slide_content, unsafe_allow_html=True)
                
                # Get or fetch related videos
                current_topic = video_topics[current_slide]
                st.write("### 🎬 Related Educational Videos")
                
                # Cache videos by topic to avoid repeated searches
                if current_topic not in st.session_state.cached_videos:
                    with st.spinner("Searching for relevant videos..."):
                        related_videos = get_related_videos(current_topic)
                        st.session_state.cached_videos[current_topic] = related_videos
                
                videos = st.session_state.cached_videos.get(current_topic, [])
                
                if videos:
                    # Display first video embedded
                    first_video = videos[0]
                    st.write(f"**{first_video['title']}**")
                    st.video(first_video['url'])
                    
                    # Show additional video options
                    if len(videos) > 1:
                        st.write("**More videos on this topic:**")
                        for i, video in enumerate(videos[1:], 1):
                            if st.button(f"Show video {i}: {video['title']}", key=f"vid_{current_slide}_{i}"):
                                st.video(video['url'])
                else:
                    st.warning("""
                    No relevant videos found. Try these educational channels:
                    - [CrashCourse](https://www.youtube.com/c/crashcourse)
                    - [Khan Academy](https://www.youtube.com/c/khanacademy)
                    - [TED-Ed](https://www.youtube.com/teded)
                    """)
                
                # SIMPLIFIED TEXT-TO-SPEECH SECTION
                st.write("### 🔊 Text-to-Speech")
                
                if st.button("🔊 Play Audio", key=f"play_{current_slide}"):
                    # Remove the title and just read the content by default
                    content_text = re.sub(r'^### Slide \d+: .*?\n', '', slide_content)
                    
                    # Check if text is too long and needs to be chunked
                    text_chunks = chunk_text_for_tts(content_text)
                    
                    # If multiple chunks, create a player for each chunk
                    if len(text_chunks) > 1:
                        st.write(f"Content divided into {len(text_chunks)} parts for better playback:")
                        for i, chunk in enumerate(text_chunks):
                            audio_data = text_to_speech(chunk, lang='en', slow=False)
                            if audio_data:
                                st.write(f"Part {i+1}:")
                                st.audio(audio_data, format='audio/mp3')
                    else:
                        # Single chunk
                        audio_data = text_to_speech(content_text, lang='en', slow=False)
                        if audio_data:
                            st.audio(audio_data, format='audio/mp3')
            else:
                # Assessment Section
                if not st.session_state.questions:
                    with st.spinner("Generating questions..."):
                        full_content = "\n".join(slides)
                        response = run_gemini_task(assessment_prompt, full_content)
                        if response:
                            question_blocks = response.strip().split("\n\n")
                            questions = []
                            for block in question_blocks:
                                lines = block.split("\n")
                                if len(lines) >= 5:
                                    question = lines[0]
                                    options = lines[1:5]
                                    correct_line = lines[5] if len(lines) > 5 else "Correct Answer: Unknown"
                                    if ": " in correct_line:  # Check if the correct answer line is valid
                                        correct_answer = correct_line.split(": ")[1].split(". ", 1)[0]
                                        explanation = correct_line.split(". ", 1)[1] if ". " in correct_line else ""
                                        questions.append({
                                            "question": question,
                                            "options": options,
                                            "correct": correct_answer,
                                            "explanation": explanation
                                        })
                            
                            # Ensure we have exactly 10 questions
                            if len(questions) < 10:
                                st.error("""
                                Failed to generate enough questions. 
                                Common reasons:
                                1. The slides content is too short
                                2. The content is not technical enough
                                3. Gemini API limitations
                                """)
                                # Reset questions for later generation
                                st.session_state.questions = []
                            else:
                                st.session_state.questions = questions
                        else:
                            st.error("Failed to generate questions.")
                
                if st.session_state.questions:
                    st.write("### Assessment Questions:")
                    answers = []
                    for i, q in enumerate(st.session_state.questions):
                        # Clean the question from extra numbering
                        question_text = q["question"].strip()
                        if question_text.startswith(str(i+1) + "."):
                            question_text = question_text[len(str(i+1)) + 1:].strip()
                        
                        # Clean the options from unwanted symbols
                        options = [opt.strip().lstrip("- ").strip() for opt in q["options"]]
                        
                        # Display the question with options
                        user_answer = st.radio(
                            f"Q{i+1}: {question_text}",  # Fixed numbering
                            options,
                            key=f"q{i}",
                            index=None  # No default option selected
                        )
                        answers.append(user_answer.split(".")[0].strip() if user_answer else None)
                    
                    if st.button("Submit"):
                        score = 0
                        results = []
                        for i, (user, q) in enumerate(zip(answers, st.session_state.questions)):
                            correct = q["correct"]
                            explanation = q["explanation"]
                            if user == correct:
                                score +=1
                                results.append({"correct": True, "explanation": ""})
                            else:
                                results.append({
                                    "correct": False,
                                    "explanation": explanation
                                })
                        
                        st.write(f"### Score: {score}/{len(answers)}")
                        
                        # Show "Try again" message if the score is less than 70%
                        if (score / len(answers)) * 100 < 70:
                            st.warning("Try again! You need to improve your performance.")
                        

                        if score < len(answers):
                            st.write("### Incorrect Answers:")
                            for i, (result, q) in enumerate(zip(results, st.session_state.questions)):
                                if not result["correct"]:
                                    # Clean the question text again to ensure no duplicate numbering
                                    question_text = q["question"].strip()
                                    if question_text.startswith(str(i+1) + "."):
                                        question_text = question_text[len(str(i+1)) + 1:].strip()
                                    
                                    # Clean the explanation to remove duplicate "Explanation:"
                                    explanation = result["explanation"].replace("Explanation:", "").strip()
                                    
                                    st.write(f"**Q{i+1}:** {question_text}")
                                    st.write(f"Your Answer: {answers[i]} | Correct: {q['correct']}")
                                    st.write(f"**Explanation:** {explanation}")
                                    st.write("---")
                else:
                    st.warning("Questions not generated yet.")
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Previous"):
                if current_slide > 0:
                    st.session_state.current_slide -=1
                    st.rerun()
        with col2:
            if st.button("Next ➡️"):
                if current_slide < len(slides) - 1:
                    st.session_state.current_slide +=1
                    st.rerun()
                else:
                    st.warning("You are on the last slide.")
        st.write(f"**Slide {current_slide+1} of {len(slides)+1}**")
else:
    st.warning("⚠️ Upload a PDF file to continue.")