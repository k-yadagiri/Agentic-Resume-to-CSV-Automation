import streamlit as st
import zipfile
import os
import tempfile
import pandas as pd

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import pdfplumber
from docx import Document

# ------------------ BASIC SETUP ------------------
load_dotenv()

st.set_page_config(
    page_title="AI-Powered Resume Analyzer & CSV Generator",
    page_icon="📄"
)

st.title("📄 AI-Powered Resume Analyzer & CSV Generator")
st.write(
    "Upload a ZIP file containing multiple resumes (PDF or DOCX). "
    "The system extracts structured information and generates a CSV file."
)

# ------------------ FILE UPLOAD ------------------
uploaded_zip = st.file_uploader(
    "Upload ZIP file with resumes",
    type=["zip"]
)

# ------------------ HELPER FUNCTIONS ------------------
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

# ------------------ MAIN LOGIC ------------------
if uploaded_zip:

    with tempfile.TemporaryDirectory() as tmpdir:

        # Save ZIP file
        zip_path = os.path.join(tmpdir, uploaded_zip.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        # Extract ZIP
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        st.success("ZIP file extracted successfully")

        # ------------------ READ ALL RESUMES (FIXED) ------------------
        resume_texts = []

        for root, _, files in os.walk(tmpdir):
            for file in files:
                file_path = os.path.join(root, file)

                if file.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)

                elif file.lower().endswith(".docx"):
                    text = extract_text_from_docx(file_path)

                else:
                    continue

                if text.strip() != "":
                    resume_texts.append(text)

        if not resume_texts:
            st.error("No valid resumes found in the ZIP file.")
            st.stop()

        st.info(f"Total resumes found: {len(resume_texts)}")

        # ------------------ BUILD SAFE BATCH PROMPT ------------------
        combined_text = ""

        for i, resume in enumerate(resume_texts, start=1):
            combined_text += f"""
====================
RESUME_ID: {i}
====================
{resume}
"""

        prompt = f"""
You are an AI resume analyzer.

You will receive multiple resumes.
Each resume starts with a RESUME_ID.

Rules:
- Process each resume independently.
- DO NOT mix information between resumes.
- DO NOT guess missing information.
- If a field is missing, write NA.

For EACH resume, output EXACTLY in this format:

RESUME_ID: <number>
Name:
Email:
Phone:
Skills:
Experience Summary:
LinkedIn:
GitHub:

Do not use JSON.
Do not add explanations.

Resumes:
{combined_text}
"""

        # ------------------ LLM CALL (SAFE) ------------------
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.2
        )

        try:
            response = model.invoke(prompt)
            output = response.content
        except Exception:
            st.error("LLM quota limit reached. Please try again later.")
            st.stop()

        # ------------------ PARSE OUTPUT ------------------
        blocks = output.split("RESUME_ID:")
        rows = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")

            data = {
                "Name": "NA",
                "Email": "NA",
                "Phone": "NA",
                "Skills": "NA",
                "Experience Summary": "NA",
                "LinkedIn": "NA",
                "GitHub": "NA",
            }

            for line in lines:
                for key in data:
                    if line.startswith(key + ":"):
                        data[key] = line.replace(key + ":", "").strip()

            rows.append(data)

        if not rows:
            st.error("Failed to extract structured resume data.")
            st.stop()

        # ------------------ CSV GENERATION ------------------
        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False)

        st.success("Resume analysis completed successfully!")

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="resume_analysis.csv",
            mime="text/csv"
        )
