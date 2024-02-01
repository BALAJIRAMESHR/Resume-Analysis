import streamlit as st
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(job_description, resume):
    content = [job_description, resume]
    cv = CountVectorizer()
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix[1][0] * 100

def main():
    st.title("Resume Matching App")

    # File upload for job description
    st.header("Upload Job Description:")
    job_description_key = "job_description_file"
    job_description_file = st.file_uploader("Choose a file (docx)", type=["docx"], key=job_description_key)
    job_description_text = ""

    if job_description_file is not None:
        job_description_text = docx2txt.process(job_description_file)
        st.success("Job description uploaded successfully!")

    # File upload for resume
    st.header("Upload Resume:")
    resume_key = "resume_file"
    resume_file = st.file_uploader("Choose a file (docx)", type=["docx"], key=resume_key)
    resume_text = ""

    if resume_file is not None:
        resume_text = docx2txt.process(resume_file)
        st.success("Resume uploaded successfully!")

    # Analysis button
    if st.button("Analyze"):
        if job_description_text and resume_text:
            similarity_percentage = calculate_similarity(job_description_text, resume_text)
            st.success(f"Resume matches the job description by: {similarity_percentage:.0f}%")
        else:
            st.warning("Please upload both job description and resume before analysis.")

if __name__ == "__main__":
    main()
