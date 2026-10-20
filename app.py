import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ERROR_TYPES,
    fetch_available_models,
    DATA_DIR,
)
from src.pipeline import analyze_answer
from src.ml_classifier import (
    predict_correctness,
    is_classifier_available,
    DEFAULT_MODEL_PATH,
    EXP_A_MODEL_PATH,
    EXP_B_MODEL_PATH,
)

st.set_page_config(
    page_title="Explainable NLP Diagnostic Student Assessment",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling for error badges
st.markdown("""
<style>
    .badge-conceptual { background-color: #6f42c1; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-methodological { background-color: #0366d6; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-procedural { background-color: #f0883e; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-computational { background-color: #d73a49; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-logical_causal { background-color: #d97706; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-question_misinterpretation { background-color: #8b5cf6; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-incomplete_reasoning { background-color: #64748b; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-none { background-color: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    
    .badge-ml-correct { background-color: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-ml-partial { background-color: #f0883e; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-ml-incorrect { background-color: #d73a49; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; }

    .step-card { border-left: 4px solid #ddd; padding: 8px 12px; margin-bottom: 8px; background: #fdfdfd; border-radius: 4px; }
    .step-card-error { border-left: 4px solid #d73a49; padding: 8px 12px; margin-bottom: 8px; background: #fff5f5; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Local Model Settings")
    st.caption("Powered by local LM Studio instance (No cloud API / No paid keys)")

    base_url_input = st.text_input("LM Studio Endpoint", value=DEFAULT_BASE_URL)

    # Fetch models from local endpoint
    models_available = fetch_available_models(base_url_input)
    if models_available:
        st.success(f"🟢 Connected ({len(models_available)} models found)")
        # Filter to prioritize chat model
        chat_models = [m for m in models_available if "embed" not in m.lower()]
        default_index = 0
        if DEFAULT_MODEL in chat_models:
            default_index = chat_models.index(DEFAULT_MODEL)
        selected_model = st.selectbox("Active Generative Model", options=chat_models or models_available, index=default_index)
    else:
        st.error("🔴 LM Studio Disconnected")
        st.info("Start LM Studio, load your model (`deepseek-r1-distill-qwen-1.5b`), and start the local server on port 1234.")
        selected_model = DEFAULT_MODEL

    st.divider()

    # Traditional ML Baseline Status & Configuration
    st.header("🤖 ML Baseline Classifier")
    ml_context_selection = st.selectbox(
        "ML Context Mode",
        options=[
            "Baseline (Answer Only)",
            "Experiment A (Question + Answer)",
            "Experiment B (Question + Ref + Answer)"
        ],
        index=0
    )

    if ml_context_selection == "Experiment A (Question + Answer)":
        active_ml_mode = "question_answer"
        active_ml_path = EXP_A_MODEL_PATH
    elif ml_context_selection == "Experiment B (Question + Ref + Answer)":
        active_ml_mode = "question_ref_answer"
        active_ml_path = EXP_B_MODEL_PATH
    else:
        active_ml_mode = "baseline"
        active_ml_path = DEFAULT_MODEL_PATH

    ml_ready = is_classifier_available(active_ml_path)
    if ml_ready:
        st.success(f"🟢 Active: `{ml_context_selection.split()[0]}` Ready")
        st.caption(f"Artifact: `{active_ml_path.name}`")
    else:
        st.warning(f"🟡 Artifact missing: `{active_ml_path.name}`")
        st.caption("Run `evaluation/train_classifier.py` to train all context variants.")

    st.divider()

    # Load Shortlist Benchmark Examples
    st.header("📚 Benchmark Examples (SAF)")
    shortlist_path = DATA_DIR / "diagnostic_shortlist.csv"
    shortlist_df = None
    sample_options = ["-- Custom (Manual Input) --"]

    if shortlist_path.exists():
        try:
            shortlist_df = pd.read_csv(shortlist_path)
            for idx, row in shortlist_df.iterrows():
                q_snippet = str(row["question"])[:40].replace("\n", " ")
                sample_options.append(f"Case {idx + 1} [Score {row['score']}]: {q_snippet}...")
        except Exception as e:
            st.warning(f"Could not load shortlist: {e}")

    def on_sample_change():
        sel = st.session_state.sample_selector
        if sel != "-- Custom (Manual Input) --" and shortlist_df is not None:
            case_num = int(sel.split()[1]) - 1
            row = shortlist_df.iloc[case_num]
            st.session_state.input_question = str(row.get("question", ""))
            st.session_state.input_reference = str(row.get("reference_answer", ""))
            st.session_state.input_rubric = "1.0 pt: Complete, accurate explanation; Deduct proportional credit for missing or flawed mechanics."
            st.session_state.input_student = str(row.get("provided_answer", ""))
            st.session_state.saf_human_score = float(row.get("score", 0.0))
            st.session_state.saf_human_feedback = str(row.get("answer_feedback", ""))

    st.selectbox("Load Pre-selected Sample", options=sample_options, key="sample_selector", on_change=on_sample_change)

    st.caption("Benchmark: Short-Answer-Feedback (SAF) Communication Networks English dataset.")

# ----------------- MAIN INTERFACE -----------------
st.title("Explainable NLP-Based Diagnostic Assessment")
st.markdown("**Distinguishing Conceptual, Methodological, Procedural, and Computational Errors in Free-Form Student Answers**")

# Input fields
col_left, col_right = st.columns(2)

with col_left:
    question_val = st.text_area(
        "Question",
        value=st.session_state.get("input_question", ""),
        height=130,
        placeholder="Enter the assessment question here..."
    )
    reference_val = st.text_area(
        "Reference Answer",
        value=st.session_state.get("input_reference", ""),
        height=130,
        placeholder="Enter the model reference solution..."
    )

with col_right:
    rubric_val = st.text_area(
        "Marking Rubric",
        value=st.session_state.get("input_rubric", ""),
        height=130,
        placeholder="e.g. 1 point for core definition, 1 point for operational condition..."
    )
    student_val = st.text_area(
        "Student Answer",
        value=st.session_state.get("input_student", ""),
        height=130,
        placeholder="Enter the student's free-form answer to diagnose..."
    )

analyze_button = st.button("🚀 Analyze Student Reasoning", type="primary", use_container_width=True)

if analyze_button:
    if not student_val.strip():
        st.warning("⚠️ Please provide a student answer to analyze.")
    elif not question_val.strip():
        st.warning("⚠️ Please provide the question text.")
    else:
        # Run ML Correctness Classifier (kept strictly independent from LLM diagnosis)
        ml_result = predict_correctness(
            student_answer=student_val,
            question=question_val,
            reference_answer=reference_val,
            model_path=active_ml_path,
            mode=active_ml_mode
        )

        with st.spinner("Analyzing reasoning steps, verifying evidence, and diagnosing errors locally..."):
            result = analyze_answer(
                question=question_val,
                reference_answer=reference_val,
                rubric=rubric_val,
                student_answer=student_val,
                base_url=base_url_input,
                model=selected_model
            )

        if result.get("status") == "error":
            st.error(f"❌ Assessment Error: {result.get('error')}")
            st.info(result.get("feedback"))
        else:
            st.success("✅ Diagnostic Assessment Complete")

            # 1. Top Metrics Cards (LLM Diagnostic Assessment)
            col1, col2, col3 = st.columns(3)

            score_val = result["score"]
            err_type = result["error_type"]
            first_step = result["first_error_step"]

            with col1:
                st.metric("Estimated Score (Rubric)", f"{score_val:.2f} / 1.0")
                st.progress(score_val)

            with col2:
                badge_class = f"badge-{err_type}"
                display_label = err_type.replace('_', ' ').capitalize()
                st.markdown(f"**Error Category**")
                st.markdown(f'<span class="{badge_class}">{display_label} Error</span>', unsafe_allow_html=True)
                st.caption(f"Taxonomy class: `{err_type}`")

            with col3:
                step_display = f"Step {first_step}" if first_step else "None (No error detected)"
                st.metric("First-Error Localization", step_display)
                if first_step:
                    st.caption("Earliest point where reasoning deviated")

            st.divider()

            # 2. Traditional ML Baseline Correctness Card (Independent)
            st.subheader("🤖 Traditional ML Baseline: Correctness Classification")
            st.caption(f"Statistical shallow baseline predicting correctness tier using `{ml_context_selection}`. Operates independently from the LLM diagnostic reasoning pipeline.")

            if not ml_result.get("available"):
                st.warning(f"⚠️ ML Model Unavailable: Artifact `{active_ml_path.name}` not found. Run `evaluation/train_classifier.py` to fit the model.")
            elif ml_result.get("error"):
                st.warning(f"⚠️ ML Classifier Note: {ml_result.get('error')}")
            else:
                pred_label = ml_result.get("predicted_label", "Unknown")
                probs = ml_result.get("probabilities", {})

                ml_col1, ml_col2 = st.columns([1, 2])
                with ml_col1:
                    badge_style = "badge-ml-correct" if pred_label == "Correct" else ("badge-ml-partial" if pred_label == "Partially correct" else "badge-ml-incorrect")
                    st.markdown("**Predicted Correctness Tier:**")
                    st.markdown(f'<span class="{badge_style}">{pred_label}</span>', unsafe_allow_html=True)
                    st.caption(f"Model: `{ml_result.get('model_type')}`")

                with ml_col2:
                    st.markdown("**Class Probability Estimates:**")
                    p_c1, p_c2, p_c3 = st.columns(3)
                    with p_c1:
                        p_corr = probs.get("Correct", 0.0)
                        st.metric("Correct", f"{p_corr * 100:.1f}%")
                        st.progress(p_corr)
                    with p_c2:
                        p_part = probs.get("Partially correct", 0.0)
                        st.metric("Partially correct", f"{p_part * 100:.1f}%")
                        st.progress(p_part)
                    with p_c3:
                        p_inc = probs.get("Incorrect", 0.0)
                        st.metric("Incorrect", f"{p_inc * 100:.1f}%")
                        st.progress(p_inc)

                st.caption(f"ℹ️ **Important Caveat:** {ml_result.get('caveat')}")

            st.divider()

            # 3. Extracted Reasoning Steps Breakdown
            st.subheader("🪜 Reasoning Steps Breakdown")
            steps = result.get("reasoning_steps", [])
            if steps:
                for step in steps:
                    s_num = step.get("step_number")
                    s_text = step.get("text")
                    is_valid = step.get("is_valid", True)
                    issue = step.get("issue")

                    is_first_err = (first_step is not None and s_num == first_step)

                    if is_first_err:
                        st.markdown(
                            f"""<div class="step-card-error">
                            <strong>❌ Step {s_num} — First Error Identified [{display_label}]:</strong><br>
                            {s_text}<br>
                            <small style="color: #c53030;"><strong>Diagnostic Issue:</strong> {result.get('first_error_description') or issue or 'Reasoning discrepancy with reference.'}</small>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    elif not is_valid:
                        st.markdown(
                            f"""<div class="step-card-error">
                            <strong>⚠️ Step {s_num} [Subsequent Error]:</strong><br>
                            {s_text}<br>
                            <small style="color: #c53030;">{issue or 'Subsequent invalid step.'}</small>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""<div class="step-card">
                            <strong>✅ Step {s_num} [Valid]:</strong><br>
                            {s_text}
                            </div>""",
                            unsafe_allow_html=True
                        )
            else:
                st.write("No distinct reasoning steps could be segmented.")

            # 4. Grounded Evidence Span
            st.subheader("🔍 Supporting Evidence")
            evidence_span = result.get("evidence_span")
            if evidence_span:
                st.markdown(f"> *\"{evidence_span}\"*")
                st.caption("🔒 Verified authentic verbatim span quoted directly from the student answer.")
            else:
                st.info("No specific localized error span (the answer is either fully correct or an omission).")

            # 5. Actionable Pedagogical Feedback
            st.subheader("💡 Targeted Actionable Feedback")
            st.info(result.get("feedback"))

            # 6. Scoring Rationale
            st.subheader("📝 Scoring Rationale")
            st.write(result.get("score_rationale"))

            # 7. DeepSeek-R1 Chain of Thought Trace
            raw_think = result.get("raw_thinking")
            if raw_think:
                with st.expander("🧠 Explainability: DeepSeek-R1 Thinking Trace (<think>)"):
                    st.markdown(f"```text\n{raw_think}\n```")

            # 8. SAF Benchmark Human Grader Comparison
            if "saf_human_score" in st.session_state:
                with st.expander("📊 SAF Ground-Truth Benchmark Comparison"):
                    comp_c1, comp_c2 = st.columns(2)
                    with comp_c1:
                        st.markdown("**Original Human Grader Assessment**")
                        st.write(f"Human Score: **{st.session_state.saf_human_score}**")
                        st.write(f"Human Grader Feedback: *{st.session_state.saf_human_feedback}*")
                    with comp_c2:
                        st.markdown("**Local NLP Model Diagnosis**")
                        st.write(f"Model Score: **{score_val}**")
                        st.write(f"Diagnosed Category: **{display_label}**")
                        st.write(f"Model Feedback: *{result.get('feedback')}*")
