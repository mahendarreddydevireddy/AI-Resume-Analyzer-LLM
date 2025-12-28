import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import pandas as pd
import zipfile
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('resumeanalyzer')

class ResumeData(TypedDict):
    summary: Annotated[str, "Provide a concise professional summary from the resume"]
    education: Annotated[str, "List education details (degrees, institutions, years)"]
    projects: Annotated[str, "Summarize key projects mentioned in the resume"]
    skills: Annotated[str, "Extract technical and soft skills from the resume"]
    experience: Annotated[str, "Extract total years of professional experience"]
    contact_details: Annotated[str, "Provide contact details (email, phone, LinkedIn if available)"]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
fm = model.with_structured_output(ResumeData)

st.title("📝 :red[Resume] :blue[TO] :green[CSV] 𝄜")

uploaded_zip = st.file_uploader("Upload a ZIP file containing resumes (PDFs)", type=["zip"])

results = {}

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]

        for pdf_file in pdf_files:
            with z.open(pdf_file) as f:
                pdf_reader = PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""

                # Pass text to Gemini model
                response = fm.invoke(text)

                # Store in dictionary with filename as key
                results[pdf_file] = response


    st.subheader("📑 Extracted Resume Data (Dictionary)")
    st.json(results)

     
    df = pd.DataFrame.from_dict(results, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "filename"}, inplace=True)

    st.subheader("📊 Resume Data (DataFrame)")
    st.dataframe(df)

     
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="resume_data.csv",
        mime="text/csv",
    )