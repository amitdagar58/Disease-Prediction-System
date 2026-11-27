"""Local Ollama-powered educational chatbot with a rule-based offline fallback."""
from __future__ import annotations
import re
import json
import os
import urllib.error
import urllib.request
from src.disease_info import DISCLAIMER, PROFILES, ALIASES, disease_report, find_profile, normalize

EMERGENCY_TERMS = [
    "can't breathe", "cannot breathe", "difficulty breathing", "severe shortness of breath",
    "chest pain", "chest pressure", "unconscious", "severe bleeding", "stroke", "face drooping",
    "slurred speech", "new weakness", "sudden weakness", "seizure", "overdose", "suicidal",
    "loss of bladder", "loss of bowel", "blue lips", "coughing blood"
]

ALGORITHM_ANSWERS = {
    "random forest": "**Random Forest** trains many decision trees using different samples/features and combines their votes. This often improves stability compared with one tree. It can model non-linear patterns, but it may be slower and less interpretable than a single tree.",
    "decision tree": "**Decision Tree** learns a sequence of feature-based questions, such as whether a symptom is present. Each branch narrows the choices until a leaf gives a class. It is easy to explain but can overfit without suitable limits.",
    "knn": "**K-Nearest Neighbors (KNN)** compares a new row with nearby training rows and predicts the class most common among its k nearest neighbors. Feature scaling matters; prediction can be slow on large datasets and results depend on k and distance.",
    "logistic regression": "**Logistic Regression** learns weights for input features and estimates class probabilities (multiclass implementations combine class scores). It is a useful baseline and often relatively interpretable, but may not capture complex non-linear patterns without feature engineering.",
    "naive bayes": "**Naive Bayes** uses Bayes' theorem to estimate class probabilities from observed features. It makes a simplifying conditional-independence assumption. It is fast, though the assumption may not match real symptom relationships.",
    "svm": "**Support Vector Machine (SVM)** learns a decision boundary that separates classes while seeking a wide margin. This project uses a linear SVM. It can work well with many features, but scaling and parameter choices matter.",
}

PROJECT_ANSWERS = {
    "project": "**Project overview:** This Streamlit Disease Prediction System turns selected symptom features into a row, then a trained classification model predicts one label from the dataset. It includes six algorithms, model metrics, dataset exploration, and this offline Q&A assistant. It is a student/educational prototype, not a clinical diagnostic tool.",
    "ai": "**AI vs ML:** Artificial Intelligence (AI) is the broader field of building systems that perform tasks associated with intelligent behavior. Machine Learning (ML) is a part of AI in which a model learns patterns from examples. In this project, the classifier is the ML component; the user interface and rule-based chatbot add interactive decision-support/educational behavior.",
    "data science": "**Data Science in this project:** The dataset is loaded and checked, symptom columns are treated as features, the disease column is the target, and the app presents class distributions and evaluation metrics. Data quality, label balance, duplicates, and representative examples affect how meaningful the results are.",
    "software engineering": "**Software Engineering in this project:** The app separates concerns into modules for data loading, model training/prediction, disease knowledge, chatbot logic, and configuration. Testing, dependency management, clear error messages, documentation, version control, and safe handling of medical disclaimers improve maintainability.",
    "metrics": "**Evaluation metrics:** Accuracy = correct predictions / all predictions. Precision measures how often predictions for a class are correct. Recall measures how many true examples of a class are found. F1-score balances precision and recall. A confusion matrix shows actual vs predicted classes. Macro averages give each class equal weight. High test scores do not prove clinical safety or real-world reliability.",
    "overfit": "**Overfitting** happens when a model learns training-set details too closely and performs worse on unseen data. A held-out test set, cross-validation, simpler models, regularization, and avoiding data leakage can help assess or reduce it.",
    "dataset": "The dataset provides symptom-feature columns and a disease-label target. Selected symptoms are represented as present/1 while unselected symptoms are absent/0 for prediction. Dataset labels are not necessarily confirmed diagnoses, and data may contain imbalance, noise, duplicates, or synthetic patterns. The app's scores should be interpreted cautiously.",
}


def _contains_any(q: str, terms) -> bool:
    return any(term in q for term in terms)


def _match_disease_in_question(question: str):
    qn = normalize(question)
    candidates = list(ALIASES.keys()) + list(PROFILES.keys())
    for name in sorted(set(candidates), key=len, reverse=True):
        nn = normalize(name)
        if nn and re.search(r"\b" + re.escape(nn) + r"\b", qn):
            return ALIASES.get(name, name)
    # Common general conditions with a specific profile or alias
    for name in ["asthma", "diabetes", "migraine", "depression", "anxiety", "gout", "pneumonia", "common cold", "spondylosis"]:
        if re.search(r"\b" + re.escape(name) + r"\b", qn):
            return name
    return None


def _answer_offline_question(question: str, current_disease: str | None = None) -> str:
    """Return a curated local answer. Used when Ollama is disabled or unavailable."""
    original = re.sub(r"\s+", " ", str(question or "")).strip()
    q = original.lower()
    if not q:
        return "Please enter a question. You can ask about a disease, relief measures, symptoms, prevention, or the project's ML algorithms."

    if _contains_any(q, EMERGENCY_TERMS):
        return ("**Please seek emergency medical help now** if these symptoms are happening currently. Contact your local emergency number or go to the nearest emergency department. Do not rely on this chatbot to assess an emergency.")

    # Explain algorithm questions before broad disease matching.
    for key, answer in ALGORITHM_ANSWERS.items():
        if key in q or (key == "knn" and "nearest neighbor" in q) or (key == "svm" and "support vector" in q) or (key == "naive bayes" and "bayes" in q):
            return answer + DISCLAIMER

    if _contains_any(q, ["what is this project", "about this project", "project overview", "explain this project", "your project"]):
        return PROJECT_ANSWERS["project"] + DISCLAIMER
    if _contains_any(q, ["software engineering", "software design", "modular", "testing", "unit test", "architecture"]):
        return PROJECT_ANSWERS["software engineering"] + DISCLAIMER
    if _contains_any(q, ["data science", "eda", "exploratory data"]):
        return PROJECT_ANSWERS["data science"] + DISCLAIMER
    if _contains_any(q, ["what is ai", "what is ml", "artificial intelligence", "machine learning", "difference between ai", "how ai", "how ml"]):
        return PROJECT_ANSWERS["ai"] + DISCLAIMER
    if _contains_any(q, ["precision", "recall", "f1", "accuracy", "confusion matrix", "evaluation metric", "metrics"]):
        return PROJECT_ANSWERS["metrics"] + DISCLAIMER
    if _contains_any(q, ["overfit", "overfitting", "underfit", "data leakage"]):
        return PROJECT_ANSWERS["overfit"] + DISCLAIMER
    if _contains_any(q, ["dataset", "features", "target column", "training data", "training set", "symptom columns"]):
        return PROJECT_ANSWERS["dataset"] + DISCLAIMER

    # Use the disease explicitly named in the question, otherwise use the latest prediction for contextual questions.
    disease = _match_disease_in_question(original)
    if disease is None and current_disease and _contains_any(q, [
        "this disease", "this condition", "this label", "my prediction", "latest prediction", "predicted disease", "the disease", "analyze my",
        "relief", "cure", "treatment", "what does it mean", "explain it", "tell me about it", "precautions"
    ]):
        disease = current_disease

    if disease:
        profile = find_profile(disease)
        if _contains_any(q, ["prevention", "prevent", "avoid", "reduce risk"]):
            if profile:
                return f"### Prevention and risk reduction for {profile['title']}\n\n{profile['prevention']}" + DISCLAIMER
        if _contains_any(q, ["relief", "feel better", "self care", "self-care", "home remedy", "at home", "manage pain", "what should i do", "instructions"]):
            if profile:
                steps = "\n".join(f"- {x}" for x in profile["relief"])
                return f"### General relief guidance: {profile['title']}\n\n{steps}\n\n**When to seek care:** {profile['care']}" + DISCLAIMER
        if _contains_any(q, ["symptom", "sign", "feel", "cause", "caused", "why", "what is", "explain", "about", "meaning", "analysis", "detail", "complete", "everything", "know"]):
            return disease_report(disease)
        if profile:
            return disease_report(disease)
        return disease_report(str(disease))

    if _contains_any(q, ["symptom", "symptoms", "signs", "what do i have", "which disease", "diagnose", "diagnosis"]):
        return ("Symptoms alone cannot reliably identify a disease because many conditions overlap. This chatbot cannot diagnose you. Record when symptoms started, severity, associated symptoms, and relevant medical history, then discuss them with a healthcare professional. If you have severe or sudden symptoms, seek urgent care." + DISCLAIMER)
    if _contains_any(q, ["medicine", "medication", "antibiotic", "dosage", "dose", "tablet", "pill", "prescribe", "prescription", "cure", "treatment"]):
        return ("Treatment depends on the confirmed condition, your medical history, allergies, age, pregnancy status, and other medicines. I can't safely recommend a specific medicine or dose. Please consult a qualified healthcare professional or pharmacist; do not use leftover prescription medicines." + DISCLAIMER)
    if _contains_any(q, ["prevent", "prevention", "healthy", "health tips"]):
        return ("General health-supporting habits include hand hygiene, appropriate vaccination, adequate sleep, balanced nutrition, regular activity suitable for your health, avoiding tobacco, and attending recommended health checks. Prevention is condition-specific, and these steps do not prevent every disease." + DISCLAIMER)
    if _contains_any(q, ["hello", "hi", "hey", "good morning", "good evening"]):
        return "Hello! I'm your offline Disease Prediction Project assistant. Ask me about your latest predicted label, general relief and prevention, symptoms, the dataset, or any of the six ML algorithms."
    if _contains_any(q, ["thank", "thanks"]):
        return "You're welcome! Remember that model labels are educational outputs, not confirmed diagnoses."
    if current_disease and _contains_any(q, ["prediction", "predicted", "last result", "last disease"]):
        return f"Your most recent model output is **{current_disease}**. This is a dataset label, not a confirmed diagnosis. Ask 'Analyze my prediction', 'What does this label mean?', or 'How can I get general relief?' to see the local information available."

    return (
        "I work offline using a local educational knowledge base, so I don't have unlimited generative knowledge. I can help with:\n\n"
        "- **Disease information:** 'Analyze my prediction' or 'What is spondylosis?'\n"
        "- **Relief and prevention:** 'How can I get relief from this condition?' or 'How can it be prevented?'\n"
        "- **ML algorithms:** 'Explain Random Forest', 'How does KNN work?', or 'Compare SVM and Logistic Regression'.\n"
        "- **Project concepts:** 'Explain the dataset', 'What are precision and recall?', or 'How is AI/ML used here?'\n\n"
        "For labels outside the local disease knowledge base, I will say when disease-specific information is unavailable instead of making it up."
        + DISCLAIMER
    )


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """You are a helpful educational assistant inside a student Disease Prediction System.
Answer the user's actual question clearly, with headings or bullet points when useful.
You can explain diseases in general, symptoms, possible causes, general self-care, prevention,
when to seek medical care, this software project, Python, Streamlit, Data Science, AI, ML,
classification algorithms, evaluation metrics, datasets, and software engineering.

MEDICAL SAFETY RULES:
- A machine-learning dataset label is not a confirmed diagnosis. Never say the user definitely has a disease.
- Do not prescribe medication, dosage, or personalized treatment. Suggest consulting a qualified clinician/pharmacist.
- Give only general, low-risk self-care advice and clearly explain uncertainty.
- If the user reports emergency warning signs such as severe breathing difficulty, severe chest pain,
  stroke signs, sudden weakness, seizure, or loss of bladder/bowel control, advise emergency care immediately.
- Do not invent specific medical facts. Say when you are unsure.
- Keep the answer useful and direct, and use simple English unless asked otherwise.
"""


def ollama_is_available() -> bool:
    """Quickly check whether the local Ollama service responds."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


def _answer_with_ollama(question: str, current_disease: str | None = None) -> str | None:
    """Call a local Ollama model. Returns None if Ollama/model is unavailable."""
    context = f"Latest ML dataset label (not a diagnosis): {current_disease}." if current_disease else "There is no saved prediction yet."
    prompt = f"Context: {context}\n\nUser question: {question}"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.3},
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        answer = str(data.get("message", {}).get("content", "")).strip()
        if answer:
            return answer + "\n\n---\n*Educational information only. A model prediction is not a confirmed diagnosis.*"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
        return None
    except Exception:
        return None
    return None


def answer_health_question(question: str, current_disease: str | None = None) -> str:
    """Use local Ollama for flexible answers, falling back to curated offline responses."""
    original = re.sub(r"\s+", " ", str(question or "")).strip()
    if not original:
        return _answer_offline_question(original, current_disease)
    # Handle urgent warning signs locally and immediately, without waiting for a model.
    if _contains_any(original.lower(), EMERGENCY_TERMS):
        return _answer_offline_question(original, current_disease)
    answer = _answer_with_ollama(original, current_disease)
    if answer:
        return answer
    fallback = _answer_offline_question(original, current_disease)
    return fallback + "\n\n*Local Ollama is not responding or the selected model is unavailable. Showing the built-in knowledge-base answer instead.*"
