import streamlit as st

st.set_page_config(page_title="Diagnostic Student Assessment", page_icon="🧠", layout="wide")
st.title("Explainable NLP-Based Diagnostic Assessment")
st.caption("Project Version 0.1 — foundation")

st.info("Pipeline: reasoning extraction → RAG → first-error localization → error classification → scoring → feedback.")

question = st.text_area("Question", height=120)
reference = st.text_area("Reference Answer", height=120)
rubric = st.text_area("Rubric", height=120)
student_answer = st.text_area("Student Answer", height=180)

if st.button("Analyze Answer"):
    if not student_answer.strip():
        st.warning("Please enter a student answer.")
    else:
        st.warning("Version 0.1 is only the foundation. Assessment modules will be added next.")
