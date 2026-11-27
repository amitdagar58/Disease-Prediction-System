import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import uuid
from src.data_utils import load_dataset, inspect_dataset
from src.modeling import load_or_train_models, predict_disease
from src.chatbot import answer_health_question, ollama_is_available, OLLAMA_MODEL
from src.disease_info import disease_report
from src.config import DATASET_PATH, PROJECT_ROOT, TEST_SIZE

HISTORY_PATH = PROJECT_ROOT / "data" / "prediction_history.json"

def load_prediction_history():
    try:
        if HISTORY_PATH.exists():
            with HISTORY_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []

def save_prediction_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = HISTORY_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    temp_path.replace(HISTORY_PATH)

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")

# Premium Clinical AI visual theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --navy: #10243A; --navy2: #173653; --teal: #0D9488;
  --teal-light: #CCFBF1; --ink: #182B3D; --muted: #64748B;
  --surface: #FFFFFF; --canvas: #F3F7FA; --line: #E3EBF1;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--canvas); }
[data-testid="stHeader"] { background: rgba(243,247,250,.88); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#10243A 0%,#173653 100%); }
[data-testid="stSidebar"] * { color: #EAF3F8 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 10px; padding: .48rem .65rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.10); }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #B8CBD8 !important; }
h1,h2,h3 { font-family: 'Manrope',sans-serif !important; color: var(--navy) !important; letter-spacing: -.035em; }
h1 { font-weight: 800 !important; }
h2 { font-weight: 750 !important; margin-top: 1.2rem !important; }
h3 { font-weight: 700 !important; }
p, label, .stMarkdown { color: #34495E; }
[data-testid="stMetric"] { background: white; border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 16px; box-shadow: 0 5px 18px rgba(16,36,58,.045); }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--navy) !important; font-family: 'Manrope',sans-serif; font-weight: 800; }
div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--line); border-radius: 16px; background: white; box-shadow: 0 5px 20px rgba(16,36,58,.035); }
.stButton > button, [data-testid="stFormSubmitButton"] > button { border-radius: 10px; font-weight: 700; min-height: 2.7rem; border: 1px solid #D6E2E9; transition: all .18s ease; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background: var(--teal); border-color: var(--teal); color: white; }
.stButton > button[kind="primary"]:hover { background: #0B7F76; border-color: #0B7F76; box-shadow: 0 5px 14px rgba(13,148,136,.2); }
[data-testid="stMultiSelect"] > div, [data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea { border-radius: 11px; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
[data-testid="stExpander"] { background: white; border: 1px solid var(--line); border-radius: 12px; margin-bottom: .55rem; }
.hero {
  background: linear-gradient(115deg,#10243A 0%,#173E59 60%,#0D7777 100%);
  border-radius: 22px; padding: 1.65rem 1.8rem; color: white;
  box-shadow: 0 12px 30px rgba(16,36,58,.13); margin-bottom: 1.25rem;
}
.hero .eyebrow { color: #9FE8DC; text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 800; margin-bottom: .45rem; }
.hero .heading { color: white; font-family: 'Manrope',sans-serif; font-size: clamp(1.55rem,3vw,2.25rem); font-weight: 800; letter-spacing: -.04em; line-height: 1.15; }
.hero .subheading { color: #D5E6ED; margin-top: .55rem; font-size: .95rem; max-width: 780px; }
.pill { display:inline-block; margin-top:.9rem; padding:.32rem .7rem; border:1px solid rgba(180,245,232,.35); border-radius:99px; color:#C5F7EB; font-size:.75rem; font-weight:700; }
.section-kicker { color: var(--teal); font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.13em; margin-bottom:.2rem; }
div[data-testid="stChatMessage"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: .8rem; }
/* High-contrast symptom selector */
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background: #243449 !important; border-color: #243449 !important;
}
[data-testid="stMultiSelect"] input,
[data-testid="stMultiSelect"] [data-baseweb="select"] *,
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
  color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
}
[data-testid="stMultiSelect"] input::placeholder { color: #E5EDF5 !important; opacity: 1 !important; }
[data-baseweb="popover"] [role="option"] { color: #182B3D !important; background: #FFFFFF !important; }
[data-baseweb="popover"] [role="option"]:hover { background: #E6F4F2 !important; }
/* Replace the left sidebar with a full-width top navigation bar */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.top-navigation-label { color: #64748B; font-size: .72rem; font-weight: 800; letter-spacing: .13em; margin: .15rem 0 .25rem; }
div[role="radiogroup"] { display: flex; flex-direction: row; flex-wrap: wrap; gap: .45rem; background: #FFFFFF; border: 1px solid #DCE7ED; border-radius: 15px; padding: .55rem; box-shadow: 0 5px 18px rgba(16,36,58,.05); }
div[role="radiogroup"] label { background: #F3F7FA; border: 1px solid #E3EBF1; border-radius: 10px; padding: .55rem .9rem; margin: 0 !important; color: #29445A !important; font-weight: 700; }
div[role="radiogroup"] label:has(input:checked) { background: #0D9488 !important; border-color: #0D9488 !important; color: #FFFFFF !important; }
div[role="radiogroup"] label:has(input:checked) * { color: #FFFFFF !important; }
@media (max-width: 768px) { .block-container { padding: 1rem .8rem 2rem; } .hero { padding: 1.2rem; border-radius: 16px; } div[role="radiogroup"] { gap: .3rem; } div[role="radiogroup"] label { padding: .45rem .6rem; } }

/* Targeted contrast fixes requested from the supplied screenshots */
/* Medical safety notice: dark background with readable white text */
[data-testid="stAlert"],
[data-testid="stAlert"] > div {
  color: #FFFFFF !important;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] *,
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] strong {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
/* Symptom selector: force visible white placeholder and selected text */
div[data-testid="stMultiSelect"] div[data-baseweb="select"],
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
  background-color: #243449 !important;
  border-color: #243449 !important;
}
div[data-testid="stMultiSelect"] div[data-baseweb="select"] input,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] input::placeholder,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] span,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] svg {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
div[data-testid="stMultiSelect"] div[data-baseweb="tag"] * {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
/* Keep dropdown choices readable on their light menu background */
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] * {
  color: #182B3D !important;
  -webkit-text-fill-color: #182B3D !important;
}
/* Dark action buttons shown in prediction history and chatbot: white labels */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:not([kind="primary"]),
[data-testid="stFormSubmitButton"] > button:not([kind="primary"]) {
  background-color: #202A36 !important;
  border-color: #344454 !important;
}
.stButton > button:not([kind="primary"]):hover,
[data-testid="stFormSubmitButton"] > button:not([kind="primary"]):hover {
  background-color: #344454 !important;
  border-color: #526579 !important;
}

</style>
<div class="hero">
  <div class="eyebrow">Clinical intelligence · Educational prototype</div>
  <div class="heading">Disease Prediction System</div>
  <div class="subheading">Explore symptom patterns with machine learning, review educational disease information, and keep your prediction history in one place.</div>
  <span class="pill">● AI-powered learning workspace</span>
</div>
""", unsafe_allow_html=True)

def section_hero(kicker, title, description):
    st.markdown(
        f'<div style="margin:.4rem 0 1.15rem 0">'
        f'<div class="section-kicker">{kicker}</div>'
        f'<div style="font-family:Manrope,sans-serif;font-size:1.65rem;font-weight:800;'
        f'letter-spacing:-.035em;color:#10243A">{title}</div>'
        f'<div style="color:#64748B;margin-top:.25rem">{description}</div></div>',
        unsafe_allow_html=True
    )

st.warning(
    "Medical safety notice: this educational prototype cannot diagnose a disease or recommend personalized treatment. "
    "A model output is a dataset label, not a confirmed diagnosis. For medical concerns, consult a qualified healthcare professional. "
    "Seek urgent care for severe or rapidly worsening symptoms."
)

@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_dataset(DATASET_PATH)

@st.cache_resource(show_spinner="Training and evaluating models once. Please wait...")
def get_trained_models(_df, target_col, feature_cols, dataset_signature):
    # _df is excluded from Streamlit's expensive argument hashing. The signature
    # changes when the dataset file changes, so cached models refresh when needed.
    return load_or_train_models(_df, target_col, tuple(feature_cols))

try:
    df, target_col, feature_cols, data_mode = get_data()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.info("Place Final_Augmented_dataset_Diseases_and_Symptoms.csv in the data folder. See README.md for setup instructions.")
    st.stop()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = load_prediction_history()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_prediction_algorithm" not in st.session_state:
    st.session_state.last_prediction_algorithm = None
if "last_prediction_symptoms" not in st.session_state:
    st.session_state.last_prediction_symptoms = []

st.markdown("<div class='top-navigation-label'>PROJECT NAVIGATION</div>", unsafe_allow_html=True)
page = st.radio(
    "NAVIGATION",
    ["Disease Prediction", "Model Performance", "Prediction History", "Dataset Overview", "Health Q&A Chatbot", "AI / ML Learning Hub"],
    horizontal=True,
    label_visibility="collapsed",
    key="top_navigation",
)
status_left, status_right = st.columns([3, 2])
with status_left:
    st.caption(f"Dataset target: `{target_col}`  ·  Input format: {data_mode}")
with status_right:
    if ollama_is_available():
        st.caption(f"🟢 Ollama connected · {OLLAMA_MODEL}")
    else:
        st.caption("🟡 Offline chatbot available · Ollama not detected")
st.divider()

if page == "Disease Prediction":
    st.subheader("Step 1 — Select your symptoms")
    st.write("Select your symptoms and click the prediction button. The backend evaluates the available algorithms and automatically uses the best-performing successfully trained model based on macro F1-score.")
    if not feature_cols:
        st.error("No usable symptom features were detected. Review the dataset format in README.md.")
        st.stop()

    feature_labels = {str(f).replace("_", " ").strip().title(): f for f in feature_cols}
    selected_labels = st.multiselect(
        "Search and select symptoms", options=list(feature_labels.keys()),
        placeholder="Type a symptom, then select it...",
        help="Start typing to search available symptom features. Select all symptoms you want the model to consider.",
        key="selected_symptom_labels",
    )
    chosen = [feature_labels[label] for label in selected_labels]

    if not chosen:
        st.info("Select one or more symptoms to enable disease prediction.")
    else:
        st.caption("Algorithm evaluation and selection run in the backend. Evaluation scores and model-selection controls are hidden from the user-facing dashboard.")
        if st.button("Predict possible disease", type="primary", width="stretch"):
            try:
                dataset_signature = f"{DATASET_PATH.stat().st_size}-{DATASET_PATH.stat().st_mtime_ns}"
                models, metrics, model_features, training_info = get_trained_models(
                    df, target_col, tuple(feature_cols), dataset_signature
                )
                valid_metrics = metrics.dropna(subset=["F1-score (macro)"])
                if valid_metrics.empty:
                    raise RuntimeError("No model produced valid evaluation scores. Check the dataset and terminal output.")
                # Backend-only model selection: choose the successfully trained model
                # with the highest macro F1-score on the held-out evaluation split.
                selected_model = str(valid_metrics.sort_values("F1-score (macro)", ascending=False).iloc[0]["Algorithm"])
                result = predict_disease(models[selected_model], model_features, chosen, data_mode)
                timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                record = {
                    "id": uuid.uuid4().hex,
                    "timestamp": timestamp,
                    "disease": str(result),
                    "algorithm": selected_model,
                    "symptoms": [str(x).replace("_", " ") for x in chosen],
                }
                st.session_state.last_prediction = str(result)
                st.session_state.last_prediction_algorithm = selected_model
                st.session_state.last_prediction_symptoms = record["symptoms"]
                st.session_state.last_prediction_record_id = record["id"]
                st.session_state.prediction_history.insert(0, record)
                save_prediction_history(st.session_state.prediction_history)
                st.session_state.show_disease_analysis = False
                st.session_state.prediction_followup_answer = ""
            except Exception as exc:
                st.error(f"Prediction could not be completed: {exc}")
                st.caption("On the first run, the backend must train and evaluate the models. If it remains here for several minutes, check the VS Code terminal for errors.")

    result = st.session_state.last_prediction
    if result:
        st.subheader("Step 3 — Prediction result")
        st.success(f"Predicted dataset label: **{result}**")
        st.caption(f"Algorithm used: {st.session_state.last_prediction_algorithm or 'Not recorded'}")
        if st.session_state.last_prediction_symptoms:
            st.write("**Selected symptoms:** " + ", ".join(st.session_state.last_prediction_symptoms))
        st.info("This is a model-generated pattern match, not a confirmed diagnosis. Symptoms can overlap, and model output can be wrong.")

        with st.container(border=True):
            st.markdown("#### 🔎 Analyze this disease")
            st.write("Review general information, common symptoms, possible contributing factors, general relief measures, prevention, and warning signs.")
            if st.button("Analyze This Disease", type="primary", width="stretch"):
                st.session_state.show_disease_analysis = True
        if st.session_state.get("show_disease_analysis", False):
            with st.container(border=True):
                st.subheader(f"Disease analysis: {result}")
                st.markdown(disease_report(result))
                st.markdown("#### Ask a follow-up question about this prediction")
                st.caption("The local chatbot may provide general health education. It cannot confirm a diagnosis or prescribe treatment.")
                with st.form("prediction_followup_form", clear_on_submit=True):
                    followup = st.text_input("Your question", placeholder=f"Ask something about {result}...")
                    submit_followup = st.form_submit_button("Ask question")
                if submit_followup:
                    if followup.strip():
                        st.session_state.chat_history.append({"role": "user", "content": followup.strip()})
                        answer = answer_health_question(followup.strip(), current_disease=result)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.session_state.prediction_followup_answer = answer
                    else:
                        st.warning("Enter a question first.")
                if st.session_state.get("prediction_followup_answer"):
                    st.markdown("**Chatbot answer**")
                    st.markdown(st.session_state.prediction_followup_answer)
        st.write("Educational next step: discuss your symptoms and concerns with a healthcare professional.")
    else:
        st.caption("Your prediction and disease analysis will appear here after you predict a label.")

    st.divider()
    st.subheader("Recent prediction history")
    st.caption("History is saved to data/prediction_history.json and remains after app restarts. You can remove individual records below.")
    history = st.session_state.prediction_history
    if not history:
        st.info("No saved predictions yet.")
    else:
        for item in history[:10]:
            with st.container(border=True):
                h1, h2 = st.columns([5, 1])
                with h1:
                    st.markdown(f"**{item.get('disease', 'Unknown label')}**")
                    st.caption(f"{item.get('timestamp', '')} · {item.get('algorithm', 'Unknown algorithm')}")
                    st.write("Symptoms: " + ", ".join(item.get("symptoms", [])))
                with h2:
                    if st.button("Remove", key=f"remove_prediction_{item.get('id')}"):
                        st.session_state.prediction_history = [x for x in history if x.get("id") != item.get("id")]
                        save_prediction_history(st.session_state.prediction_history)
                        st.rerun()

elif page == "Model Performance":
    st.subheader("Model Accuracy")
    st.write("View the accuracy of the selected model and compare the accuracy of each evaluated algorithm.")

    try:
        dataset_signature = f"{DATASET_PATH.stat().st_size}-{DATASET_PATH.stat().st_mtime_ns}"
        with st.spinner("Loading saved model accuracy or evaluating the algorithms for the first time..."):
            models, metrics, model_features, training_info = get_trained_models(
                df, target_col, tuple(feature_cols), dataset_signature
            )

        valid_metrics = metrics.dropna(subset=["Accuracy"]).copy()
        if valid_metrics.empty:
            st.error("Accuracy is not available. Check the dataset and terminal output.")
        else:
            # The prediction workflow selects its model using the existing macro-F1 rule.
            # This page intentionally displays accuracy only, without other metrics or metadata.
            if "F1-score (macro)" in valid_metrics.columns:
                selected_rows = valid_metrics.dropna(subset=["F1-score (macro)"])
                if not selected_rows.empty:
                    selected_row = selected_rows.sort_values(
                        "F1-score (macro)", ascending=False
                    ).iloc[0]
                else:
                    selected_row = valid_metrics.sort_values("Accuracy", ascending=False).iloc[0]
            else:
                selected_row = valid_metrics.sort_values("Accuracy", ascending=False).iloc[0]

            st.markdown("### Selected model accuracy")
            st.metric("Accuracy", f"{selected_row['Accuracy'] * 100:.2f}%")
            st.caption(f"Algorithm: {selected_row['Algorithm']}")

            st.markdown("### Algorithm accuracy comparison")
            accuracy_table = valid_metrics[["Algorithm", "Accuracy"]].copy()
            accuracy_table["Accuracy"] = accuracy_table["Accuracy"].map(
                lambda value: f"{value * 100:.2f}%" if pd.notna(value) else "N/A"
            )
            table_html = accuracy_table.to_html(
                index=False,
                escape=True,
                border=0,
                classes="model-accuracy-table",
                justify="left",
            )
            st.markdown(
                """
                <style>
                .model-accuracy-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
                .model-accuracy-table th, .model-accuracy-table td { padding: 0.7rem 0.8rem; border-bottom: 1px solid rgba(128,128,128,.25); text-align: left; }
                .model-accuracy-table th { font-weight: 700; background: rgba(128,128,128,.10); }
                </style>
                """ + table_html,
                unsafe_allow_html=True,
            )
            st.caption("Accuracy is measured on the held-out test data. It is a dataset evaluation score, not a guarantee of clinical accuracy.")
    except Exception as exc:
        st.error(f"Model accuracy could not be calculated: {exc}")
        st.caption("The first evaluation may take some time. After successful training, saved model metrics can be reused.")

elif page == "Prediction History":
    st.subheader("Saved prediction history")
    st.write("Saved records are stored locally and remain available after restarting Streamlit. Remove a record only when you choose to.")
    history = st.session_state.prediction_history
    if not history:
        st.info("No saved predictions yet. Make a prediction on the Disease Prediction page first.")
    else:
        for item in history:
            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(f"**{item.get('disease', 'Unknown label')}**")
                    st.caption(f"{item.get('timestamp', '')} · Algorithm: {item.get('algorithm', 'Unknown')}")
                    st.write("Symptoms: " + ", ".join(item.get("symptoms", [])))
                with right:
                    if st.button("Remove", key=f"history_remove_{item.get('id')}"):
                        st.session_state.prediction_history = [x for x in history if x.get("id") != item.get("id")]
                        save_prediction_history(st.session_state.prediction_history)
                        st.rerun()
        if st.button("Clear all prediction history", type="secondary"):
            st.session_state.show_clear_history_confirm = True
        if st.session_state.get("show_clear_history_confirm", False):
            st.warning("This will permanently remove all saved prediction records from this local history file.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm clear history", type="primary"):
                    st.session_state.prediction_history = []
                    save_prediction_history([])
                    st.session_state.show_clear_history_confirm = False
                    st.rerun()
            with cancel_col:
                if st.button("Cancel"):
                    st.session_state.show_clear_history_confirm = False
                    st.rerun()

elif page == "Dataset Overview":
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Disease classes", f"{df[target_col].nunique():,}")
    st.write(f"**Target column:** `{target_col}`")
    st.write("**Detected symptom/features:**", ", ".join(map(str, feature_cols)) if feature_cols else "None")
    st.dataframe(inspect_dataset(df, target_col), width="stretch", hide_index=True)
    st.subheader("Sample rows")
    st.dataframe(df.head(10), width="stretch")
    st.subheader("Most frequent disease labels")
    st.bar_chart(df[target_col].astype(str).value_counts().head(20))
    st.caption("Dataset frequencies reflect this dataset, not real-world disease prevalence.")

elif page == "Health Q&A Chatbot":
    st.subheader("Local AI Health Q&A Chatbot")
    st.write("Ask follow-up questions about your prediction, general health education, AI/ML, Python, the dataset, or software engineering.")
    if ollama_is_available():
        st.caption(f"Local model: {OLLAMA_MODEL}. Messages are sent to Ollama on this computer; no paid API is used.")
    else:
        st.caption("Ollama is currently unavailable. The built-in rule-based knowledge base will answer supported questions until Ollama starts.")
    current_disease = st.session_state.last_prediction
    if current_disease:
        st.info(f"Current prediction context: **{current_disease}**. Ask 'Analyze my prediction' or 'How can I get relief from this condition?'")
        if st.button("Analyze my latest prediction", key="chat_analyze_latest"):
            st.session_state.chat_history.append({"role": "user", "content": "Analyze my latest prediction."})
            st.session_state.chat_history.append({"role": "assistant", "content": answer_health_question("Analyze my latest prediction.", current_disease=current_disease)})
            st.rerun()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input("Ask about a disease, relief, prevention, AI/ML, or this project...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        response = answer_health_question(question, current_disease=st.session_state.last_prediction)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.subheader("AI / ML Learning Hub")
    st.write("Short explanations for the concepts implemented in this project.")
    with st.expander("How the prediction pipeline works", expanded=True):
        st.markdown("1. **Input:** you select symptoms.\n2. **Feature row:** selected symptoms are marked present; other symptom features are marked absent.\n3. **Preprocessing:** missing values are handled, categorical values are encoded, and numeric features are scaled where needed.\n4. **Classification:** the selected trained algorithm predicts a dataset label.\n5. **Interpretation:** the app displays the label and a separate local knowledge-base report.\n6. **Safety:** the prediction is not a diagnosis and should not guide treatment without professional advice. The backend evaluates all available algorithms and automatically selects the best successfully trained model by macro F1-score; evaluation metrics and selection controls are not shown on the user-facing prediction page.")
    for name, description in [
        ("Decision Tree", "Learns feature-based decision rules in a tree. Easy to explain, but may overfit."),
        ("Random Forest", "Combines votes from many decision trees to improve stability and capture varied patterns."),
        ("KNN", "Predicts from the labels of nearby training examples; feature scaling and the choice of k matter."),
        ("Logistic Regression", "Learns class scores/probabilities from weighted features; often a useful baseline."),
        ("Naive Bayes", "Uses Bayes' theorem with a simplifying assumption that features are conditionally independent."),
        ("SVM", "Learns a boundary separating classes; this project uses a linear SVM."),
    ]:
        with st.expander(name):
            st.write(description)
    with st.expander("Software engineering concepts"):
        st.markdown("- **Modularity:** separate files for data, modelling, disease knowledge, and chatbot logic.\n- **Caching:** avoid reloading data and retraining models on every UI interaction.\n- **Testing:** verify emergency handling, offline answers, and algorithm explanations.\n- **Usability:** navigation, clear errors, prediction context, and a visible safety notice.\n- **Maintainability:** dependency list, README, and version-control-friendly structure.")
    with st.expander("What metrics mean"):
        st.markdown("- **Accuracy:** share of all predictions that are correct.\n- **Precision:** among predictions for a class, how many were correct.\n- **Recall:** among true examples of a class, how many were found.\n- **F1-score:** balances precision and recall.\n- **Confusion matrix:** counts actual vs predicted classes.\n- **Important limitation:** high scores on this dataset do not prove medical reliability.")
