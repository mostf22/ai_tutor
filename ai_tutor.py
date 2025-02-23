import os
import re
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai

# Configure Google Gemini API
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to clean text by removing HTML tags, extra spaces, and empty lines
def clean_text(text):
    """Clean the text by removing HTML tags, extra spaces, and empty lines."""
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    lines = [line.strip() for line in text.split('\n') if line.strip()]  # Remove empty lines
    return '\n'.join(lines)

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF pages, ignoring pages with no text."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n--- PAGE BREAK ---\n\n"  # Add page break for better structure
    return text.strip()

# Function to split text into paragraphs while preserving context
def split_text(text, min_chunk_size=6000, max_chunk_size=12000):
    """Split text into chunks based on paragraphs, merging small paragraphs."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += '\n\n' + para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to run Gemini task with error handling
def run_gemini_task(prompt, text):
    """Send text to Gemini model and handle the response."""
    try:
        response = model.generate_content(f"{prompt}\n\n{text}")
        return response.text
    except Exception as e:
        st.error(f"Error processing text with Gemini: {e}")
        return ""

# Improved Prompt for structuring the learning experience
explain_prompt = """
### **Transforming Educational Content into an Interactive Learning Experience**

Your task is to convert raw educational content into a structured, engaging, and interactive learning experience. Follow these guidelines to ensure the output is clear, concise, and easy to understand.

✅ **Your Responsibilities:**
1️⃣ **Divide the content into clear sections with meaningful headings.** Each section should focus on a specific topic or concept.
2️⃣ **Use structured formatting** such as bullet points, numbered lists, and highlights to improve readability.
3️⃣ **Add interactive elements** such as questions, examples, or real-world applications to make the content more engaging.
4️⃣ **Do not leave any slides empty.** Ensure every slide contains meaningful content.
5️⃣ **Extract a clear and concise title for each slide.** The title should summarize the main idea of the slide.
6️⃣ **Use diverse teaching methods** such as practical examples, interactive questions, or real-world scenarios to explain concepts.
7️⃣ **Do not include any images or visual elements.** Focus solely on text-based content.
8️⃣ **Provide detailed explanations** for each concept, ensuring that the content is thorough and comprehensive.
9️⃣ **Ensure each slide contains at least 1000-1500 words** to make the content more detailed and informative.

🎯 **Example Output Format:**
### **Section 1: Introduction to Machine Learning**
Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data and make predictions or decisions without being explicitly programmed. It is widely used in various applications such as image recognition, natural language processing, and recommendation systems.

**Detailed Explanation:** Machine learning algorithms are designed to learn patterns from data. These algorithms can be categorized into different types based on how they learn. The most common types are supervised learning, unsupervised learning, and reinforcement learning. Each type has its own set of algorithms and use cases, making machine learning a versatile tool in various industries.

**Example:** Imagine you want to build a system that can recognize handwritten digits. Using machine learning, you can train a model to recognize these digits by feeding it thousands of labeled images of handwritten numbers. The model learns the patterns in the images and can then predict the correct digit for new, unseen images.

---

### **Section 2: Types of Machine Learning**
There are three main types of machine learning:
- **Supervised Learning:** The model is trained on labeled data, where the input and output are known. Examples include regression and classification tasks.
- **Unsupervised Learning:** The model is trained on unlabeled data, and it tries to find patterns or structures in the data. Examples include clustering and dimensionality reduction.
- **Reinforcement Learning:** The model learns by interacting with an environment and receiving feedback in the form of rewards or penalties. Examples include game playing and robotics.

**Detailed Explanation:** Supervised learning is often used when the goal is to predict an outcome based on input data. Unsupervised learning is useful for discovering hidden patterns or groupings in data. Reinforcement learning is ideal for scenarios where an agent needs to learn how to interact with an environment to achieve a goal.

**Interactive Question:** Can you think of a real-world scenario where unsupervised learning might be useful?

---

### **Section 3: Applications of Machine Learning**
Machine learning is used in various fields, including:
- **Healthcare:** Predicting disease outbreaks, personalized medicine, and medical imaging analysis.
- **Finance:** Fraud detection, algorithmic trading, and risk assessment.
- **Technology:** Autonomous vehicles, natural language processing, and recommendation systems.

**Detailed Explanation:** In healthcare, machine learning models are used to analyze medical images and detect diseases like cancer at an early stage, improving patient outcomes. In finance, machine learning algorithms are used to detect fraudulent transactions by identifying unusual patterns in transaction data. In technology, machine learning powers recommendation systems that suggest products or content based on user behavior.

**Real-World Scenario:** In healthcare, machine learning models are used to analyze medical images and detect diseases like cancer at an early stage, improving patient outcomes.

---

**Input Text:**
"""

# Improved Prompt for generating questions
assessment_prompt = """
Generate 10 high-quality multiple-choice questions (MCQ) based on the following educational content. 
Each question should be clear, concise, and directly related to the key concepts in the content. 
Follow these guidelines to ensure the questions are effective:

✅ **Guidelines for Questions:**
1️⃣ **Focus on Key Concepts:** Each question should test the understanding of an important concept from the content.
2️⃣ **Clear and Concise:** Questions should be easy to understand and free from ambiguity.
3️⃣ **Balanced Difficulty:** Include a mix of easy, medium, and slightly challenging questions.
4️⃣ **Realistic Options:** Provide 4 options for each question, with only one correct answer. The incorrect options should be plausible but clearly wrong.
5️⃣ **Avoid Trick Questions:** Ensure that the questions are fair and test knowledge, not the ability to decipher tricky wording.
6️⃣ **Cover Diverse Topics:** Ensure that the questions cover a wide range of topics from the content.
7️⃣ **Correct Answer Format:** Clearly indicate the correct answer for each question using the format "Correct Answer: X".

🎯 **Example Format:**
1. What is the primary goal of supervised learning?
   A. To find hidden patterns in unlabeled data
   B. To predict outcomes based on labeled data
   C. To learn by interacting with an environment
   D. To reduce the dimensionality of data
   Correct Answer: B

2. Which of the following is an example of unsupervised learning?
   A. Predicting house prices based on historical data
   B. Grouping customers based on purchasing behavior
   C. Training a robot to navigate a maze
   D. Classifying emails as spam or not spam
   Correct Answer: B

Continue for 10 questions.

**Input Text:**
"""

# Streamlit UI
st.title("📚 AI Explainer Model")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Your File", type="pdf")

# Initialize session state
if "slides" not in st.session_state:
    st.session_state.slides = []

if "current_slide" not in st.session_state:
    st.session_state.current_slide = 0

if "titles" not in st.session_state:
    st.session_state.titles = []

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        lesson_text = extract_text_from_pdf(uploaded_file)
        lesson_text = clean_text(lesson_text)
    
    st.write("📜 **Extracted Content (First 500 chars):**")
    st.write(lesson_text[:500] + "...")
    
    if st.button("🔍 Explain Lesson"):
        with st.spinner("Processing with AI..."):
            try:
                # Split text into chunks if necessary
                if len(lesson_text) > 1000:
                    text_chunks = split_text(lesson_text)
                    presentation = ""
                    for chunk in text_chunks:
                        response = run_gemini_task(explain_prompt, chunk)
                        if response.strip():  # Ensure the response is not empty
                            presentation += response + "\n---\n"
                else:
                    presentation = run_gemini_task(explain_prompt, lesson_text)
                
                if presentation:
                    slides = presentation.split("---")
                    cleaned_slides = []
                    titles = []
                    for slide in slides:
                        slide = slide.strip()
                        # Ensure the slide is not empty and contains meaningful content
                        if slide and not re.match(r"^\s*$", slide) and not re.match(r"^\s*[-|]+\s*$", slide):
                            # Extract title from the first line of the slide
                            title = slide.split('\n')[0].strip()
                            titles.append(title)
                            # Add the title back to the slide content
                            slide_content = f" {title}\n\n" + '\n'.join(slide.split('\n')[1:])
                            cleaned_slides.append(slide_content)
                    st.session_state.slides = cleaned_slides
                    st.session_state.titles = titles
                    st.session_state.current_slide = 0
                    st.success("Lesson explained successfully!")
                else:
                    st.error("No presentation generated!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if "slides" in st.session_state and st.session_state.slides:
        slides = st.session_state.slides
        titles = st.session_state.titles
        current_slide = st.session_state.current_slide

        # Display the table of contents
        st.sidebar.title("📑 Table of Contents")
        for i, title in enumerate(titles):
            if st.sidebar.button(f"{i + 1}. {title}"):
                st.session_state.current_slide = i
                st.rerun()

        # Add assessment button to the table of contents
        if st.sidebar.button(f"{len(titles) + 1}. Assessment"):
            st.session_state.current_slide = len(slides)  # Move to the assessment section
            st.rerun()

        # Display the current slide
        with st.expander("📖 Click to view slide content", expanded=True):
            if current_slide < len(slides):
                slide_content = slides[current_slide]
                st.markdown(slide_content, unsafe_allow_html=True)
            else:
                st.markdown("## 📝 Assessment")
                if "questions" not in st.session_state:
                    with st.spinner("Generating MCQ questions..."):
                        try:
                            # Use the full content to generate questions
                            full_content = "\n".join(st.session_state.slides)
                            response = run_gemini_task(assessment_prompt, full_content)
                            if response.strip():  # Ensure the model returned questions
                                st.session_state.questions = response
                            else:
                                st.error("Failed to generate questions. Please try again.")
                                st.session_state.questions = ""
                        except Exception as e:
                            st.error(f"Error generating questions: {e}")
                            st.session_state.questions = ""

                if "questions" in st.session_state and st.session_state.questions:
                    st.write("### MCQ Questions:")
                    questions = st.session_state.questions.strip().split("\n\n")
                    st.session_state.correct_answers = []
                    user_answers = []

                    for i, question_block in enumerate(questions):
                        lines = question_block.split("\n")
                        if len(lines) >= 6:  # Ensure there are enough lines for a valid question
                            question = lines[0]
                            options = lines[1:5]
                            correct_answer_line = lines[5]
                            if ": " in correct_answer_line:
                                correct_answer = correct_answer_line.split(": ")[1]
                            else:
                                correct_answer = "Unknown"  # If correct answer is not found

                            st.write(f"**{question}**")
                            user_answer = st.radio(f"Select an answer for question {i + 1}:", options, key=f"q{i}")
                            user_answers.append(user_answer)
                            st.session_state.correct_answers.append(correct_answer)
                        else:
                            st.error(f"Invalid question format for question {i + 1}.")

                    if st.button("Submit Answers"):
                        score = 0
                        wrong_answers = []  # لتخزين الأسئلة التي تمت الإجابة عليها بشكل خاطئ

                        for i, (user_answer, correct_answer) in enumerate(zip(user_answers, st.session_state.correct_answers)):
                            if user_answer.startswith(correct_answer):
                                score += 1
                            else:
                                wrong_answers.append((i + 1, user_answer, correct_answer))  # حفظ السؤال الخطأ

                        st.write(f"### Your Score: {score} out of {len(questions)}")
                        st.success("Thank you for completing the assessment!")

                        # عرض الإجابات الصحيحة أسفل كل سؤال تمت الإجابة عليه بشكل خاطئ
                        if wrong_answers:
                            st.write("### Correct Answers for Wrongly Answered Questions:")
                            for question_num, user_answer, correct_answer in wrong_answers:
                                st.write(f"**Question {question_num}:**")
                                st.write(f"Your answer: {user_answer}")
                                st.write(f"**Correct answer:** {correct_answer}")
                                st.write("---")  # إضافة فاصل بين الأسئلة
                else:
                    st.warning("No questions generated. Please ensure the content is sufficient and try again.")

            st.markdown(
                """
                <style>
                .stExpander {
                    max-height: 500px;
                    overflow-y: auto;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Previous"):
                if current_slide > 0:
                    st.session_state.current_slide -= 1
                    st.rerun()
        with col2:
            if st.button("Next ➡️"):
                if current_slide < len(slides):
                    st.session_state.current_slide += 1
                    st.rerun()
        st.write(f"**Slide {current_slide + 1} of {len(slides) + 1}**")  # +1 to include the assessment section
else:
    st.warning("⚠️ **No file uploaded!**")