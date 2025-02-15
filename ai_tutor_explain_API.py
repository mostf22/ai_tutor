from fastapi import FastAPI, UploadFile, File
import os
import re
import pdfplumber
import google.generativeai as genai

app = FastAPI()

# إعداد Google Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFlv5cQ6kSoAUSse9YIak0W10NFuoMZ_Q"  
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# وظيفة لتنظيف النص من علامات HTML
def clean_text(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# وظيفة لاستخراج النص من ملف PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Prompts
content_processing_prompt = """
### ** Comprehensive Educational Content Analysis**

You are a highly skilled educational content analyst. Your role is to extract, organize, and present key insights in a structured and engaging manner.

✅ **Your Objectives:**
1️⃣ **Identify and categorize the core topics with precision.**
2️⃣ **Summarize complex concepts into easy-to-understand explanations.**
3️⃣ **Enhance understanding by providing real-world applications and case studies.**
4️⃣ **Format content using Markdown for structured readability.**
5️⃣ **Emphasize key points with bullet points and concise lists.**
6️⃣ **Include thought-provoking questions to promote critical thinking.**

🎯 **Example Output Format:**
### **🔍 The Role of AI in Healthcare**
- **Definition:** AI technologies used to improve medical diagnosis and treatment.
- **Example:** Machine learning algorithms detecting cancer with high accuracy.
- **Key Benefit:** Faster diagnoses and improved patient outcomes.

💡 **Reflection Question:** How can AI revolutionize early disease detection?

**Input Text:**
"""

evaluation_prompt = """
### ** Clarity & Readability Enhancement**

You are an expert in evaluating and refining educational content. Your goal is to assess the clarity, readability, and engagement level of the given text.

✅ **Your Tasks:**
1️⃣ **Determine the target audience proficiency level (Beginner, Intermediate, Advanced) and justify your choice.**
2️⃣ **Identify unclear sections and provide actionable improvement suggestions.**
3️⃣ **Recommend techniques to make the content more engaging (e.g., analogies, case studies).**
4️⃣ **Assess the logical flow and coherence, suggesting ways to enhance it.**

🎯 **Example Output Format:**
🔹 **Reading Level:** Intermediate
🔹 **Issues Identified:**
   - Some complex terms need simplification.
   - The second section lacks logical flow and should be reorganized.
🔹 **Proposed Enhancements:**
   - Integrate real-life case studies to reinforce concepts.
   - Use visuals or step-by-step explanations to improve comprehension.

**Input Text:**
"""

caption_prompt = """
### ** Crafting Engaging & Concise Captions**

You specialize in distilling key takeaways into impactful, attention-grabbing captions that summarize the essence of the content effectively.

✅ **Your Tasks:**
1️⃣ **Extract the top 3-5 core ideas from the text.**
2️⃣ **Condense them into short, compelling captions (8-12 words each).**
3️⃣ **Use creative and engaging language, incorporating emojis where appropriate.**

🎯 **Example Output Format:**
 " AI: Shaping the Future of Medical Diagnosis!"
 " The Power of Data in Smart Decision Making!"
 " Can Machines Think Like Humans?"

**Input Text:**
"""

presentation_prompt = """
### ** Structuring an Interactive Learning Experience**

Your expertise lies in transforming raw educational content into a well-structured, interactive, and engaging learning experience.

✅ **Your Responsibilities:**
1️⃣ **Segment the content into clearly defined sections with meaningful headings.**
2️⃣ **Use structured formatting (bullet points, tables, highlights) for better readability.**
3️⃣ **Incorporate interactive elements such as thought-provoking discussion prompts.**
4️⃣ **Summarize critical takeaways in a closing section.**

🎯 **Example Output Format:**
### ** Understanding Neural Networks**
🔹 **What is a Neural Network?**
   - A computational model inspired by the human brain.
   - Used in speech recognition, medical imaging, and financial forecasting.

🔹 **Key Concepts:**
   - **Neurons:** Fundamental processing units mimicking biological neurons.
   - **Deep Learning:** Leveraging vast datasets for enhanced accuracy.

💡 **Discussion Prompt:** How do neural networks improve user experience in AI applications?

**Input Text:**
"""

# تنفيذ المهام باستخدام Gemini
def run_gemini_task(prompt, text):
    response = model.generate_content(f"{prompt}\n\n{text}")
    return response.text

@app.post("/analyze_pdf/")
async def analyze_pdf(file: UploadFile = File(...)):
    pdf_text = extract_text_from_pdf(file.file)
    key_insights = run_gemini_task(content_processing_prompt, pdf_text)
    clarity_analysis = run_gemini_task(evaluation_prompt, pdf_text)
    captions = run_gemini_task(caption_prompt, pdf_text)
    presentation = run_gemini_task(presentation_prompt, pdf_text)
    
    return {
        "key_insights": key_insights,
        "clarity_analysis": clarity_analysis,
        "captions": captions,
        "presentation": presentation
    }

# تشغيل التطبيق محليًا عند استخدام Railway
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
