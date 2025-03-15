import os
import re
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts
import asyncio
from io import BytesIO
from youtube_dl import YoutubeDL
import tempfile
import hashlib
import random

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

def answer_student_question(slide_content, question, allow_out_of_scope=False):
    """Generate an answer to a student's question about the slide."""
    try:
        # Choose the appropriate prompt based on whether out-of-scope answers are allowed
        if allow_out_of_scope:
            qa_prompt = f"""
            You are an expert educational tutor helping a student understand complex topics. 
            Answer the following question thoroughly but concisely, even if it's outside the scope of the current slide content.
            
            Your response should:
            1. Directly address the student's question with accurate information
            2. Use clear, simple language appropriate for educational purposes
            3. Include relevant examples when helpful for understanding
            4. Connect the answer to broader concepts where appropriate
            5. Highlight key terms or concepts using bold formatting
            6. If the question is related to the slide content, prioritize that information
            7. If the question is outside the scope of the slide, provide a helpful answer based on general knowledge
            8. End with a brief check for understanding if the concept is complex
            
            SLIDE CONTENT (for reference):
            {slide_content}
            
            STUDENT'S QUESTION:
            {question}
            
            Remember to provide a helpful response regardless of whether the question relates directly to the slide content.
            """
        else:
            qa_prompt = f"""
            You are an expert educational tutor helping a student understand complex topics. 
            Based on the slide content provided below, answer the student's question thoroughly but concisely.
            
            Your response should:
            1. Directly address the student's question with accurate information from the slide
            2. Use clear, simple language appropriate for educational purposes
            3. Include relevant examples when helpful for understanding
            4. Connect the answer to broader concepts where appropriate
            5. Highlight key terms or concepts using bold formatting
            6. Prioritize information from the slide content, but supplement with general knowledge when necessary
            7. End with a brief check for understanding if the concept is complex
            8. If the question requires knowledge beyond the slide content, indicate this clearly
            
            SLIDE CONTENT:
            {slide_content}
            
            STUDENT'S QUESTION:
            {question}
            
            Remember to balance depth and clarity in your response. If the question requires knowledge beyond the slide content, indicate this clearly.
            """
        
        response = model.generate_content(qa_prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't generate an answer due to an error: {e}"
    
def is_valid_youtube_url(url):
    """Check if the YouTube URL is valid using youtube-dl."""
    ydl = YoutubeDL()
    try:
        info = ydl.extract_info(url, download=False)
        return True, info.get('title', 'No title available')
    except:
        return False, None

def get_related_videos(topic, slide_content=None, max_results=3):
    """Get related YouTube videos for a given topic with improved relevance."""
    # Extract key concepts from slide content if available
    key_concepts = []
    if slide_content:
        # Extract bullet points which often contain key concepts
        bullet_points = re.findall(r'- (.*?)(?:\n|$)', slide_content)
        key_concepts = [point.strip() for point in bullet_points if len(point.strip()) > 3]
    
    # Combine slide title topic with key concepts for better search
    search_terms = [topic]
    if key_concepts:
        search_terms.extend(key_concepts[:2])  # Add up to 2 key concepts to avoid too specific queries
    
    search_query = " ".join(search_terms) + " educational video"
    
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
        'ignoreerrors': True,
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        try:
            # Get more results than needed to allow for filtering
            search_results = ydl.extract_info(f"ytsearch{max_results * 2}:{search_query}", download=False)
            if search_results and 'entries' in search_results:
                videos = []
                for entry in search_results.get('entries', []):
                    if entry:
                        # Check if video title contains any of the key terms to ensure relevance
                        title = entry.get('title', '').lower()
                        is_relevant = any(term.lower() in title for term in search_terms if len(term) > 3)
                        
                        if is_relevant or not key_concepts:  # If no key concepts or video is relevant
                            videos.append({
                                'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                                'title': entry.get('title', 'No title'),
                                'relevance_score': sum(term.lower() in title for term in search_terms)
                            })
                
                # Sort by relevance and limit to max_results
                videos.sort(key=lambda x: x['relevance_score'], reverse=True)
                return videos[:max_results]
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
def get_tts_cache_key(text, voice, rate):
    """Generate a unique key for caching TTS audio."""
    key_content = f"{text}_{voice}_{rate}"
    return hashlib.md5(key_content.encode()).hexdigest()

# Improved TTS function with caching
async def text_to_speech_async(text, voice="en-US-ChristopherNeural", rate="+0%"):
    """Convert text to speech using Edge TTS with caching."""
    # Create a cache if it doesn't exist
    if "tts_cache" not in st.session_state:
        st.session_state.tts_cache = {}
    
    # Clean and prepare text
    cleaned_text = prepare_text_for_tts(text)
    
    # Create a cache key
    cache_key = get_tts_cache_key(cleaned_text, voice, rate)
    
    # Check if we already have this audio in cache
    if cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key]
    
    # If not in cache, generate new audio
    try:
        communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate)
        
        # Use a temporary file instead of BytesIO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = temp_file.name
        
        # Save the audio to the temporary file
        await communicate.save(temp_path)
        
        # Read the audio data from the file
        with open(temp_path, 'rb') as f:
            audio_data = f.read()
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Store in cache
        st.session_state.tts_cache[cache_key] = audio_data
        
        return audio_data
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# 3. أضف دالة غير متزامنة لاستدعاء الدالة المتزامنة
def text_to_speech(text, voice="ar-SA-HamedNeural", rate="+0%"):
    """Non-async wrapper for text_to_speech_async function."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(text_to_speech_async(text, voice, rate))
    finally:
        loop.close()


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

# Function to generate motivational messages based on score
def get_motivational_message(percentage_score):
    """Return a motivational message based on the student's score percentage."""
    if percentage_score >= 70:
        # Messages for scores above 70%
        high_score_messages = [
            "🌟 Excellent work! Your hard work and study efforts are truly paying off!",
            "🎉 Amazing job! You've demonstrated a strong understanding of the material!",
            "👏 Impressive result! Keep up this excellent performance!",
            "✨ Outstanding! You've mastered most of the key concepts!",
            "🏆 Great achievement! Your dedication to learning is evident in your score!",
            "💯 Fantastic performance! You're well on your way to mastering this subject!"
        ]
        return random.choice(high_score_messages)
    else:
        # Messages for scores below 70%
        low_score_messages = [
            "💪 You're making progress! Another attempt will help solidify these concepts.",
            "🔄 Learning is a journey! Try again with what you've learned so far.",
            "🌱 Every attempt helps you grow! Review the material and give it another try.",
            "📚 You've started building a foundation! Let's strengthen it with another attempt.",
            "🧩 You're putting the pieces together! A review and second attempt will help complete the picture.",
            "🚀 You're on the right path! Another attempt will help boost your understanding."
        ]
        return random.choice(low_score_messages)

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
    st.session_state.qa_history = {}  # New: Store Q&A history for each slide
    st.session_state.assessment_attempted = False  # Track if assessment was attempted

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
                st.session_state.qa_history = {}  # Reset Q&A history
                st.session_state.assessment_attempted = False  # Reset assessment attempts
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
                st.session_state.assessment_attempted = False  # Reset when entering assessment
                st.rerun()
            else:
                st.warning("No slides available to take assessment.")

        # Content Display
        if current_slide < len(slides):
            # ---------- 1. SLIDE CONTENT (First) ----------
            with st.expander("📖 Slide Content", expanded=True):
                slide_content = slides[current_slide]
                st.markdown(slide_content, unsafe_allow_html=True)
            
            # ---------- 2. AUDIO VERSION (Second) ----------
            with st.expander("🔊 Audio Version", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    voice = st.selectbox(
                        "Choose the voice:",
                        [
                            "en-US-ChristopherNeural", # صوت إنجليزي ذكوري
                            "en-US-JennyNeural",     # صوت إنجليزي أنثوي
                        ],
                        key=f"voice_{current_slide}"
                    )
                with col2:
                    rate = st.select_slider(
                        "Speed ​​of sound:",
                        options=["-50%", "-25%", "+0%", "+25%", "+50%"],
                        value="+0%",
                        key=f"rate_{current_slide}"
                    )
                    
                if st.button("🔊 Play Audio", key=f"play_{current_slide}"):
                    # Remove the title and just read the content by default
                    content_text = re.sub(r'^### Slide \d+: .*?\n', '', slide_content)
                    
                    # Check if text is too long and needs to be chunked
                    text_chunks = chunk_text_for_tts(content_text)
                    
                    # If multiple chunks, create a player for each chunk
                    if len(text_chunks) > 1:
                        st.write(f"Content divided into {len(text_chunks)} parts for better playback:")
                        for i, chunk in enumerate(text_chunks):
                            audio_data = text_to_speech(chunk, voice=voice, rate=rate)
                            if audio_data:
                                st.write(f"Part {i+1}:")
                                st.audio(audio_data, format='audio/mp3')
                    else:
                        # Single chunk
                        audio_data = text_to_speech(content_text, voice=voice, rate=rate)
                        if audio_data:
                            st.audio(audio_data, format='audio/mp3')
            
            # ---------- 3. RELATED VIDEOS (Third) ----------
            with st.expander("🎬 Related Educational Videos", expanded=False):
                current_topic = video_topics[current_slide]
                
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
            
            # ---------- 4. Q&A SECTION (Fourth) ----------
            with st.expander("❓ Questions & Answers", expanded=False):
                # Initialize Q&A history for this slide if it doesn't exist
                slide_id = f"slide_{current_slide}"
                if slide_id not in st.session_state.qa_history:
                    st.session_state.qa_history[slide_id] = []
                
                # Display previous Q&A for this slide
                if st.session_state.qa_history[slide_id]:
                    st.write("### Previous Questions")
                    for q_a in st.session_state.qa_history[slide_id]:
                        with st.chat_message("user"):
                            st.markdown(q_a["question"])
                        with st.chat_message("assistant"):
                            st.markdown(q_a["answer"])
                
                # Input for new questions
                st.write("### Ask a New Question")
                with st.form(key=f"qa_form_{current_slide}"):
                    user_question = st.text_area("Ask a question about this slide:", key=f"q_input_{current_slide}")
                    
                    # Add checkbox for out-of-scope questions
                    out_of_scope = st.checkbox("Answer even if question is outside slide scope", key=f"out_of_scope_{current_slide}")
                    
                    submit_question = st.form_submit_button("Ask")
                
                if submit_question and user_question:
                    # Generate answer using Gemini
                    with st.spinner("Generating answer..."):
                        if out_of_scope:
                            # Modify the prompt for out-of-scope questions
                            general_prompt = f"""
                            You are an expert educational tutor helping a student understand complex topics. 
                            Answer the following question with your best knowledge, even if it's outside the scope of the current slide.
                            
                            If the question relates to the slide content, prioritize that information, but feel free to provide general knowledge as needed.
                            
                            SLIDE CONTENT (for reference):
                            {slides[current_slide]}
                            
                            STUDENT'S QUESTION:
                            {user_question}
                            
                            Please provide a helpful, educational response regardless of whether the question is directly related to the slide content.
                            """
                            answer = run_gemini_task(general_prompt, "")
                        else:
                            # Standard in-scope answer
                            answer = answer_student_question(slides[current_slide], user_question)
                        
                        # Save to history
                        st.session_state.qa_history[slide_id].append({
                            "question": user_question,
                            "answer": answer
                        })
                    
                    # Display the new Q&A
                    with st.chat_message("user"):
                        st.markdown(user_question)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
            
            # Navigation
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous"):
                    if current_slide > 0:
                        st.session_state.current_slide -= 1
                        st.rerun()
            with col2:
                if st.button("Next ➡️"):
                    if current_slide < len(slides) - 1:
                        st.session_state.current_slide += 1
                        st.rerun()
                    else:
                        st.warning("You are on the last slide.")
            st.write(f"**Slide {current_slide+1} of {len(slides)+1}**")
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
                
                # Store answers in a specific key for assessment
                if "assessment_answers" not in st.session_state:
                    st.session_state.assessment_answers = [None] * len(st.session_state.questions)
                
                answers = st.session_state.assessment_answers
                
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
                        index=None if answers[i] is None else options.index(answers[i])
                    )
                    
                    if user_answer:
                        answers[i] = user_answer
                
                # Check if assessment has been attempted and score is below 70%
                if st.session_state.assessment_attempted and st.session_state.last_score_percentage < 70:
                    button_text = "Try Again"
                else:
                    button_text = "Submit"
                
                if st.button(button_text):
                    # Validate that all questions have an answer
                    if None in answers:
                        st.warning("Please answer all questions before submitting.")
                    else:
                        # Process the selected answers (extract letter from options)
                        processed_answers = [ans.split(".")[0].strip() if ans else None for ans in answers]
                        
                        score = 0
                        results = []
                        for i, (user, q) in enumerate(zip(processed_answers, st.session_state.questions)):
                            correct = q["correct"]
                            explanation = q["explanation"]
                            if user == correct:
                                score += 1
                                results.append({"correct": True, "explanation": ""})
                            else:
                                results.append({
                                    "correct": False,
                                    "explanation": explanation
                                })
                        
                        st.write(f"### Score: {score}/{len(processed_answers)}")
                        
                        # Calculate percentage score
                        percentage_score = (score / len(processed_answers)) * 100
                        st.session_state.last_score_percentage = percentage_score
                        st.session_state.assessment_attempted = True
                        
                        # Display appropriate motivational message based on score
                        motivational_message = get_motivational_message(percentage_score)
                        st.write(f"### {motivational_message}")
                        
                        if score < len(processed_answers):
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
                                    st.write(f"Your Answer: {processed_answers[i]} | Correct: {q['correct']}")
                                    st.write(f"**Explanation:** {explanation}")
                                    st.write("---")
                        
            else:
                st.warning("Questions not generated yet.")
else:
    st.warning("⚠️ Upload a PDF file to continue.")