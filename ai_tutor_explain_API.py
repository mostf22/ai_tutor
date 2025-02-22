import os
import re
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
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
def split_text(text, min_chunk_size=1000, max_chunk_size=2000):
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
    return chunks  # Remove the limit on the number of chunks

# Function to run Gemini task with error handling
def run_gemini_task(prompt, text):
    """Send text to Gemini model and handle the response."""
    try:
        response = model.generate_content(f"{prompt}\n\n{text}")
        return response.text
    except Exception as e:
        st.error(f"Error processing text with Gemini: {e}")
        return ""

# Function to parse and fix tables using pandas
def fix_table_format(slide):
    """Attempt to parse and fix table formatting using pandas."""
    try:
        table_pattern = re.compile(r'(\|.*\|\n)+')
        match = table_pattern.search(slide)
        if match:
            table_str = match.group()
            df = pd.read_csv(StringIO(table_str), sep='|', engine='python')
            df = df.dropna(axis=1, how='all')  # Drop empty columns
            df.columns = df.columns.str.strip()
            return df.to_markdown(index=False)
        else:
            return slide
    except Exception as e:
        st.warning(f"Could not parse table: {e}")
        return "### Table removed (invalid format)\n**Note:** The system could not fix the table. Please check the table format in the original file."

# Improved Prompt for structuring the learning experience
explain_prompt = """
### **Transforming Educational Content into an Interactive Learning Experience**

Your task is to convert raw educational content into a structured, engaging, and interactive learning experience. Follow these guidelines to ensure the output is clear, concise, and easy to understand.

✅ **Your Responsibilities:**
1️⃣ **Divide the content into clear sections with meaningful headings.** Each section should focus on a specific topic or concept.
2️⃣ **Use structured formatting** such as bullet points, numbered lists, tables, and highlights to improve readability.
3️⃣ **Ensure all tables are complete and properly formatted.** If tables are missing or incomplete, reconstruct them based on the context.
4️⃣ **Summarize key takeaways** at the end of each section to reinforce learning.
5️⃣ **Add interactive elements** such as questions, examples, or real-world applications to make the content more engaging.
6️⃣ **Do not leave any slides empty.** Ensure every slide contains meaningful content.

🎯 **Example Output Format:**
### **Section 1: Introduction to Neural Networks**
Neural networks are computational models inspired by the human brain. They are widely used in various applications such as speech recognition, medical imaging, and financial forecasting. These networks consist of layers of interconnected nodes (neurons) that process data. The ability of neural networks to learn complex patterns from large datasets makes them powerful tools for tasks like image and speech recognition.

---

### **Section 2: Key Components of Neural Networks**
Neural networks consist of the following key components:
- **Input Layer:** Receives the initial data.
- **Hidden Layers:** Process the data through weighted connections.
- **Output Layer:** Produces the final result.

---

### **Section 3: Applications of Neural Networks**
Neural networks are used in various fields, including:
- **Healthcare:** Medical diagnosis, drug discovery, and patient outcome prediction.
- **Finance:** Fraud detection, risk assessment, and algorithmic trading.
- **Technology:** Autonomous vehicles, natural language processing, and recommendation systems.

---

### **Section 4: Summary**
Neural networks are versatile tools that have revolutionized many industries. Their ability to learn from data makes them essential for solving complex problems in fields like healthcare, finance, and technology. As the amount of data continues to grow, the importance of neural networks in driving innovation and efficiency will only increase.

---

### **Section 5: Common Machine Learning Algorithms**
| Algorithm        | Type           | Description                                                                 | Applications                          |
|------------------|----------------|-----------------------------------------------------------------------------|---------------------------------------|
| Neural Networks  | Supervised     | Simulates the human brain with linked processing nodes to recognize patterns. | Natural language translation, image/speech recognition, image creation. |
| Linear Regression| Supervised     | Predicts numerical values based on a linear relationship between values.    | Predicting house prices based on historical data. |
| Logistic Regression| Supervised   | Makes predictions for categorical response variables (e.g., yes/no).        | Classifying spam, quality control on a production line. |
| Clustering       | Unsupervised   | Identifies patterns in data for grouping.                                   | Identifying differences between data items that humans have overlooked. |
| Decision Trees   | Both           | Uses a branching sequence of linked decisions for prediction or classification. | Predicting numerical values (regression) and classifying data into categories. |
| Random Forests   | Supervised     | Combines the results from multiple decision trees to predict a value or category. | Improves prediction accuracy over single decision trees. |

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
                    for slide in slides:
                        slide = slide.strip()
                        # Ensure the slide is not empty and contains meaningful content
                        if slide and not re.match(r"^\s*$", slide) and not re.match(r"^\s*[-|]+\s*$", slide):
                            if "|" in slide:
                                fixed_slide = fix_table_format(slide)
                                cleaned_slides.append(fixed_slide)
                            else:
                                cleaned_slides.append(slide)
                    st.session_state.slides = cleaned_slides
                    st.session_state.current_slide = 0
                    st.success("Lesson explained successfully!")
                else:
                    st.error("No presentation generated!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if "slides" in st.session_state and st.session_state.slides:
        slides = st.session_state.slides
        current_slide = st.session_state.current_slide
        st.markdown(f"<h2 style='text-align: center; color: blue;'>Slide {current_slide + 1}</h2>", unsafe_allow_html=True)
        
        with st.expander("📖 Click to view slide content", expanded=True):
            st.markdown(slides[current_slide], unsafe_allow_html=True)
        
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
        st.write(f"**Slide {current_slide + 1} of {len(slides)}**")
else:
    st.warning("⚠️ **No file uploaded!**")