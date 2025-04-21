# Imports
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
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')




# Prompts
explain_prompt = """
### **Transform Educational Content into Interactive Slides**
Create slides appropriate for {level} learners. Follow these guidelines:

{level_instructions}

Format each slide with:
- Title starting with "### Slide X: [Title]"
- Content matching the expertise level
- Include a topic for video recommendations in a separate line in the format "VIDEO_TOPIC: topic keywords" (This line will be removed from the displayed content)

**Example Format:**
### Slide 1: [Title]
[Level-appropriate content]

VIDEO_TOPIC: [topic]

---

**Input Text:**
"""

LEVEL_INSTRUCTIONS = {
    "Basic": """
    BASIC LEVEL REQUIREMENTS:
    - Explain concepts like teaching to complete beginners
    - Use simple language and short sentences
    - Add multiple examples for each concept
    - Include definitions for technical terms
    - Break complex ideas into step-by-step explanations
    - Add "Key Point" boxes for important concepts
    """,
    
    "Intermediate": """
    INTERMEDIATE LEVEL REQUIREMENTS:
    - Balance depth and accessibility
    - Assume basic domain knowledge
    - Use technical terms with brief explanations
    - Include 1-2 examples per complex concept
    - Highlight connections between concepts
    """,
    
    "Advanced": """
    ADVANCED LEVEL REQUIREMENTS:
    - Professional, concise academic tone
    - Assume strong domain knowledge
    - Use technical jargon appropriately
    - Focus on complex relationships between concepts
    - Include case studies/research references
    """
}

quiz_prompt = """
Generate mixed question types (MCQs and True/False) based on slide count. Follow these rules:
- Create 1-3 questions per slide
- Mix question types naturally
- Follow formats:

MCQ Format:
1. [Question]
   A. [Option]
   B. [Option]
   C. [Option]
   D. [Option]
   Correct Answer: [Letter]. Explanation: [Context from slides]

True/False Format:
2. [Statement]
   A. True
   B. False
   Correct Answer: [A/B]. Explanation: [Context from slides]

Include explanations referencing specific slides. Ensure unambiguous answers.

**Example:**
1. What is supervised learning?
   A. Learning without labels
   B. Learning with labeled data
   C. Learning from rewards
   D. Learning from random patterns
   Correct Answer: B. Explanation: Defined in Slide 2 as...

2. Deep learning requires labeled data.
   A. True
   B. False
   Correct Answer: B. Explanation: Slide 3 shows...

**Input Text:**
"""


# Utility Functions Module
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





# AI Services Module
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
        level = st.session_state.get('expertise_level', 'Intermediate')

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
            9. Tailor the answer for {level} level learners
            10. Match the complexity to the user's selected expertise level

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
            9. Tailor the answer for {level} level learners
            10. Match the complexity to the user's selected expertise level

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
    



# TTS Module
def get_tts_cache_key(text, voice, rate):
    """Generate a unique key for caching TTS audio."""
    key_content = f"{text}_{voice}_{rate}"
    return hashlib.md5(key_content.encode()).hexdigest()

async def text_to_speech_async(text, voice="en-US-ChristopherNeural", rate="+0%"):
    """Convert text to speech using Edge TTS with caching."""
    # Create a cache if it doesn't exist
    if "tts_cache" not in st.session_state:
        st.session_state.tts_cache = {}
    
    # Clean and prepare text
    cleaned_text = prepare_text_for_tts(text)
    
    # Create a cache key
    cache_key = get_tts_cache_key(cleaned_text, voice, rate)
    
    # Check cache
    if cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key]
    
    # Generate new audio if not in cache
    try:
        communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = temp_file.name
        
        await communicate.save(temp_path)
        
        with open(temp_path, 'rb') as f:
            audio_data = f.read()
        
        os.unlink(temp_path)
        st.session_state.tts_cache[cache_key] = audio_data
        return audio_data
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def text_to_speech(text, voice="en-US-ChristopherNeural", rate="+0%"):
    """Non-async wrapper for text_to_speech_async function."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(text_to_speech_async(text, voice, rate))
    finally:
        loop.close()

def chunk_text_for_tts(text, max_chars=1000):
    """Split text into smaller chunks for TTS processing."""
    cleaned_text = prepare_text_for_tts(text)
    if len(cleaned_text) <= max_chars:
        return [cleaned_text]
    
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks





# Video Module
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
    # Extract key concepts from slide content
    key_concepts = []
    if slide_content:
        bullet_points = re.findall(r'- (.*?)(?:\n|$)', slide_content)
        key_concepts = [point.strip() for point in bullet_points if len(point.strip()) > 3]
    
    # Combine search terms
    search_terms = [topic]
    if key_concepts:
        search_terms.extend(key_concepts[:2])
    search_query = " ".join(search_terms) + " educational video"
    
    # YouTubeDL configuration
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
        'ignoreerrors': True,
    }
    
    # Execute search
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch{max_results * 2}:{search_query}", download=False)
            if search_results and 'entries' in search_results:
                videos = []
                for entry in search_results.get('entries', []):
                    if entry:
                        title = entry.get('title', '').lower()
                        is_relevant = any(term.lower() in title for term in search_terms if len(term) > 3)
                        
                        if is_relevant or not key_concepts:
                            videos.append({
                                'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                                'title': entry.get('title', 'No title'),
                                'relevance_score': sum(term.lower() in title for term in search_terms)
                            })
                
                # Sort and return results
                videos.sort(key=lambda x: x['relevance_score'], reverse=True)
                return videos[:max_results]
        except Exception as e:
            st.error(f"Error finding related videos: {e}")
    return []




# Quiz Module
def get_motivational_message(percentage_score):
    """Return a motivational message based on the student's score percentage."""
    if percentage_score >= 70:
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
        low_score_messages = [
            "💪 You're making progress! Another attempt will help solidify these concepts.",
            "🔄 Learning is a journey! Try again with what you've learned so far.",
            "🌱 Every attempt helps you grow! Review the material and give it another try.",
            "📚 You've started building a foundation! Let's strengthen it with another attempt.",
            "🧩 You're putting the pieces together! A review and second attempt will help complete the picture.",
            "🚀 You're on the right path! Another attempt will help boost your understanding."
        ]
        return random.choice(low_score_messages)

def parse_quiz_response(response):
    """Parse Gemini's quiz response into structured questions."""
    question_blocks = response.strip().split("\n\n")
    questions = []
    
    for block in question_blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 4:  # Minimum lines for a valid question
            try:
                question = lines[0]
                options = lines[1:-1]
                answer_line = lines[-1]
                
                if answer_line.startswith("Correct Answer:"):
                    answer_parts = answer_line.split(". ", 1)
                    correct_answer = answer_parts[0].split(": ")[1]
                    explanation = answer_parts[1] if len(answer_parts) > 1 else ""
                    
                    questions.append({
                        "question": question,
                        "options": options,
                        "correct": correct_answer,
                        "explanation": explanation
                    })
            except Exception as e:
                st.error(f"Error parsing question: {str(e)}")
    return questions




# Main Streamlit App
def main():
    st.title("📚 AI Tutor Model")

    # File uploader
    uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

    # Expertise level selector
    expertise_level = st.selectbox(
        "🎚️ Select Expertise Level",
        ["Basic", "Intermediate", "Advanced"],
        index=1,
        key="expertise_level"
    )

    # Initialize session state
    session_defaults = {
        "slides": [],
        "titles": [],
        "current_slide": 0,
        "questions": [],
        "correct_answers": [],
        "video_topics": [],
        "cached_videos": {},
        "tts_cache": {},
        "qa_history": {},
        "quiz_attempted": False,
        "quiz_answers": []
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if uploaded_file:
        with st.spinner("Extracting text..."):
            lesson_text = extract_text_from_pdf(uploaded_file)
            lesson_text = clean_text(lesson_text)
        
        if st.button("Generate"):
            try:
                level = st.session_state.expertise_level
                level_instructions = LEVEL_INSTRUCTIONS[level]
                dynamic_prompt = explain_prompt.format(
                    level=level,
                    level_instructions=level_instructions
                )
                
                if len(lesson_text) > 1000:
                    chunks = split_text(lesson_text)
                    presentation = ""
                    for chunk in chunks:
                        response = run_gemini_task(dynamic_prompt, chunk)
                        presentation += f"{response}\n---\n" if response else ""
                else:
                    presentation = run_gemini_task(dynamic_prompt, lesson_text)

                if presentation:
                    slides = [s.strip() for s in presentation.split("---") if s.strip()]
                    titles = []
                    video_topics = []
                    cleaned_slides = []
                    
                    for slide in slides:
                        title_match = re.search(r'### Slide \d+: (.+)', slide)
                        title = title_match.group(1).strip() if title_match else "Untitled Slide"
                        titles.append(title)
                        
                        topic_match = re.search(r'VIDEO_TOPIC: (.+?)$', slide, re.MULTILINE)
                        topic = topic_match.group(1).strip() if topic_match else title
                        video_topics.append(topic)
                        
                        cleaned_slide = re.sub(r'VIDEO_TOPIC: (.+?)$', '', slide, flags=re.MULTILINE)
                        cleaned_slides.append(cleaned_slide.strip())
                    
                    st.session_state.update({
                        "slides": cleaned_slides,
                        "titles": titles,
                        "video_topics": video_topics,
                        "cached_videos": {},
                        "tts_cache": {},
                        "qa_history": {},
                        "quiz_attempted": False
                    })
                    st.success("Slides generated successfully!")
                else:
                    st.error("No content generated. Check your input file.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        if st.session_state.slides:
            current_slide = st.session_state.current_slide
            slides = st.session_state.slides
            titles = st.session_state.titles
            video_topics = st.session_state.video_topics

            # Sidebar Navigation
            st.sidebar.title("📑 Table of Contents")
            for i, title in enumerate(titles):
                if st.sidebar.button(f"{i+1}. {title}"):
                    st.session_state.current_slide = i
                    st.rerun()
            
            if st.sidebar.button(f"{len(titles)+1}. Take A Quiz"):
                if len(slides) > 0:
                    st.session_state.current_slide = len(slides)
                    st.session_state.quiz_attempted = False
                    st.rerun()
                else:
                    st.warning("No slides available to take a quiz.")

            # Slide Display Logic
            if current_slide < len(slides):
                display_slide_content(current_slide, slides, video_topics)
            else:
                display_quiz_section()

    else:
        st.warning("⚠️ Upload a PDF file to continue.")

def display_slide_content(current_slide, slides, video_topics):
    """Display slide content with associated features."""
    with st.expander("📖 Slide Content", expanded=True):
        slide_content = slides[current_slide]
        st.markdown(slide_content, unsafe_allow_html=True)
    
    display_audio_version(current_slide, slide_content)
    display_related_videos(current_slide, video_topics)
    display_qa_section(current_slide, slide_content)
    handle_navigation(current_slide, len(slides))

def display_audio_version(current_slide, slide_content):
    """Display TTS controls and audio player."""
    with st.expander("🔊 Audio Version", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            voice = st.selectbox(
                "Choose the voice:",
                ["en-US-ChristopherNeural", "en-US-JennyNeural"],
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
            content_text = re.sub(r'^### Slide \d+: .*?\n', '', slide_content)
            text_chunks = chunk_text_for_tts(content_text)
            
            if len(text_chunks) > 1:
                st.write(f"Content divided into {len(text_chunks)} parts:")
                for i, chunk in enumerate(text_chunks):
                    audio_data = text_to_speech(chunk, voice=voice, rate=rate)
                    if audio_data:
                        st.write(f"Part {i+1}:")
                        st.audio(audio_data, format='audio/mp3')
            else:
                audio_data = text_to_speech(content_text, voice=voice, rate=rate)
                if audio_data:
                    st.audio(audio_data, format='audio/mp3')

def display_related_videos(current_slide, video_topics):
    """Display related YouTube videos."""
    with st.expander("🎬 Related Educational Videos", expanded=False):
        current_topic = video_topics[current_slide]
        
        if current_topic not in st.session_state.cached_videos:
            with st.spinner("Searching for relevant videos..."):
                related_videos = get_related_videos(current_topic)
                st.session_state.cached_videos[current_topic] = related_videos
        
        videos = st.session_state.cached_videos.get(current_topic, [])
        
        if videos:
            first_video = videos[0]
            st.write(f"**{first_video['title']}**")
            st.video(first_video['url'])
            
            if len(videos) > 1:
                st.write("**More videos on this topic:**")
                for i, video in enumerate(videos[1:], 1):
                    if st.button(f"Show video {i}: {video['title']}", key=f"vid_{current_slide}_{i}"):
                        st.video(video['url'])
        else:
            st.warning("No relevant videos found. Try these educational channels...")

def display_qa_section(current_slide, slide_content):
    """Display Q&A interface."""
    with st.expander("❓ Questions & Answers", expanded=False):
        slide_id = f"slide_{current_slide}"
        if slide_id not in st.session_state.qa_history:
            st.session_state.qa_history[slide_id] = []
        
        if st.session_state.qa_history[slide_id]:
            st.write("### Previous Questions")
            for q_a in st.session_state.qa_history[slide_id]:
                with st.chat_message("user"):
                    st.markdown(q_a["question"])
                with st.chat_message("assistant"):
                    st.markdown(q_a["answer"])
        
        with st.form(key=f"qa_form_{current_slide}"):
            user_question = st.text_area("Ask a question about this slide:", key=f"q_input_{current_slide}")
            out_of_scope = st.checkbox("Answer even if question is outside slide scope", key=f"out_of_scope_{current_slide}")
            submit_question = st.form_submit_button("Ask")
        
        if submit_question and user_question:
            with st.spinner("Generating answer..."):
                if out_of_scope:
                    general_prompt = f"""
                    You are an expert educational tutor. Answer the question thoroughly, even if outside slide content.
                    SLIDE CONTENT (for reference):
                    {slide_content}
                    STUDENT'S QUESTION:
                    {user_question}
                    """
                    answer = run_gemini_task(general_prompt, "")
                else:
                    answer = answer_student_question(slide_content, user_question)
                
                st.session_state.qa_history[slide_id].append({
                    "question": user_question,
                    "answer": answer
                })
                
                with st.chat_message("user"):
                    st.markdown(user_question)
                with st.chat_message("assistant"):
                    st.markdown(answer)

def handle_navigation(current_slide, total_slides):
    """Handle slide navigation controls."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous"):
            if current_slide > 0:
                st.session_state.current_slide -= 1
                st.rerun()
    with col2:
        if st.button("Next ➡️"):
            if current_slide < total_slides - 1:
                st.session_state.current_slide += 1
                st.rerun()
            else:
                st.warning("You are on the last slide.")
    st.write(f"**Slide {current_slide+1} of {total_slides+1}**")

def display_quiz_section():
    """Display quiz interface and results."""
    if not st.session_state.questions:
        with st.spinner("Generating questions..."):
            full_content = "\n".join(st.session_state.slides)
            response = run_gemini_task(quiz_prompt, full_content)
            if response:
                question_blocks = response.strip().split("\n\n")
                questions = []
                
                for block in question_blocks:
                    lines = [line.strip() for line in block.split("\n") if line.strip()]
                    if len(lines) >= 4:  # Minimum lines for a valid question
                        try:
                            question = lines[0]
                            options = lines[1:-1]
                            answer_line = lines[-1]
                            
                            if answer_line.startswith("Correct Answer:"):
                                answer_parts = answer_line.split(". ", 1)
                                correct_answer = answer_parts[0].split(": ")[1]
                                explanation = answer_parts[1] if len(answer_parts) > 1 else ""
                                
                                questions.append({
                                    "question": question,
                                    "options": options,
                                    "correct": correct_answer,
                                    "explanation": explanation
                                })
                        except Exception as e:
                            st.error(f"Error parsing question: {str(e)}")
                
                # Initialize quiz answers with proper length
                st.session_state.questions = questions
                st.session_state.quiz_answers = [None] * len(questions)  # Add this line
            else:
                st.error("Failed to generate questions. Content might be too short.")

    if st.session_state.questions:
        total_questions = len(st.session_state.questions)
        st.write(f"### Quiz ({total_questions} Questions)")
        
        # Initialize quiz_answers if not properly set
        if len(st.session_state.quiz_answers) != total_questions:
            st.session_state.quiz_answers = [None] * total_questions  # Add this line

        # Display questions
        for i, q in enumerate(st.session_state.questions):
            question_text = q["question"].split(". ", 1)[-1]
            options = [opt for opt in q["options"] if opt]
            
            if len(options) not in [2, 4]:
                continue
            
            user_answer = st.radio(
                f"Q{i+1}: {question_text}",
                options,
                key=f"q{i}",
                index=None
            )
            if user_answer:
                # Ensure safe assignment
                if i < len(st.session_state.quiz_answers):
                    st.session_state.quiz_answers[i] = user_answer
                else:
                    st.error("Question index out of range. Please regenerate the quiz.")

        
        # Handle quiz submission
        button_text = "Try Again" if st.session_state.quiz_attempted else "Submit"
        if st.button(button_text):
            if None in st.session_state.quiz_answers:
                st.warning("Please answer all questions.")
            else:
                processed_answers = [ans.split(".")[0].strip() if ans else None for ans in st.session_state.quiz_answers]
                score = sum(1 for user, q in zip(processed_answers, st.session_state.questions) if user == q["correct"])
                percentage_score = (score / total_questions) * 100
                
                st.session_state.quiz_attempted = True
                st.session_state.last_score_percentage = percentage_score
                
                st.write(f"### Score: {score}/{total_questions}")
                st.write(f"### {get_motivational_message(percentage_score)}")
                
                if score < total_questions:
                    st.write("### Incorrect Answers:")
                    for i, (user, q) in enumerate(zip(processed_answers, st.session_state.questions)):
                        if user != q["correct"]:
                            display_question_feedback(i, q, user)

def display_question_feedback(index, question, user_answer):
    """Display detailed feedback for incorrect answers."""
    question_text = question["question"].strip()
    if question_text.startswith(str(index+1) + "."):
        question_text = question_text[len(str(index+1)) + 1:].strip()
    
    explanation = question["explanation"].replace("Explanation:", "").strip()
    
    st.write(f"**Q{index+1}:** {question_text}")
    st.write(f"Your Answer: {user_answer} | Correct: {question['correct']}")
    st.write(f"**Explanation:** {explanation}")
    st.write("---")

if __name__ == "__main__":
    main()