# Disease Prediction System Using Multiple Classification Algorithms

A BTech educational project demonstrating **Artificial Intelligence (AI), Machine Learning (ML), Data Science (DS), and Software Engineering (SE)** through a Streamlit application.

## Features
- Symptom-based disease-label prediction using a backend-selected classifier
- Six ML algorithms: Decision Tree, Random Forest, KNN, Logistic Regression, Naive Bayes, and SVM
- Model comparison using accuracy, macro precision, macro recall, and macro F1-score
- Dataset overview, missing-value/duplicate summary, and disease distribution chart
- Local Ollama-powered conversational chatbot with a built-in rule-based fallback; no paid API credits required
- “Analyze This Disease” panel under predictions with available disease overview, symptoms, possible causes, general relief, prevention, and warning signs
- Current prediction is passed into chatbot context so follow-up questions like “How can I get relief from this condition?” are contextualized
- AI/ML Learning Hub explaining the six algorithms, evaluation metrics, pipeline, and software-engineering choices
- Modular Python structure and simple test suite

## Medical safety
This is an educational prototype, not a clinically validated medical device. Predictions can be incorrect and must not be used to diagnose, rule out, or treat disease. The chatbot provides general educational information only; it does not prescribe medication or personalized treatment. Seek a qualified healthcare professional for medical concerns and emergency care for severe symptoms.

## Requirements
- Python 3.10–3.12 recommended
- pip
- Windows, macOS, or Linux

## Optional local AI chatbot with Ollama (Windows)
The app uses Ollama running on your own computer for flexible, local language-model answers. The built-in knowledge-base chatbot remains available if Ollama is stopped.

1. Install Ollama for Windows from https://ollama.com/download/windows if it is not already installed.
2. Open PowerShell and verify the installation:

```powershell
ollama --version
```

3. Download a small model (recommended starting point):

```powershell
ollama pull qwen2.5:3b
```

If that model is too slow for your computer, try `ollama pull llama3.2:3b` and change the model setting below. Keep Ollama running; its local API normally listens at `http://localhost:11434`.

4. Verify the model is installed and run a quick test:

```powershell
ollama list
ollama run qwen2.5:3b
```

Type a test question, then type `/bye` to exit the interactive model. If `ollama list` shows a different model name, use that exact name in the configuration.

5. (Optional) Set environment variables in the same PowerShell terminal before launching Streamlit:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

The defaults are already these values, so this step is only needed if you choose another model or host.

## Setup (Windows PowerShell)
1. Extract the ZIP file.
2. Open the extracted `disease_prediction_system` folder in VS Code.
3. Confirm the CSV exists at `data/Final_Augmented_dataset_Diseases_and_Symptoms.csv`.
4. Open the VS Code terminal and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser. To stop the server, press `Ctrl+C` in the terminal.

## Dataset format
The app tries to detect a target column whose name contains `disease`, `prognosis`, `diagnosis`, `target`, or `label`. If no match exists, it assumes the final column is the target. Other columns are treated as features. A single symptom-list column with comma/semicolon/pipe-separated symptoms is expanded into binary features.

For the intended symptom-selection screen, the most suitable format is one row per example, a disease label column, and one column per symptom with values such as 0/1 or No/Yes. If your CSV uses a different layout, review the detected target and feature columns in the sidebar and adjust `src/data_utils.py` if needed.

## Project structure
```text
disease_prediction_system/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Final_Augmented_dataset_Diseases_and_Symptoms.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── modeling.py
│   └── chatbot.py
└── tests/
    └── test_chatbot.py
```

## How the subjects are implemented
- **AI:** interactive rule-based Q&A flow, disease-information retrieval, prediction-aware follow-up, understandable output, and safety-oriented responses. It does not use a paid generative AI API.
- **ML:** six supervised classification algorithms and held-out test-set evaluation.
- **DS:** dataset inspection, preprocessing, feature handling, disease distribution, and metric visualizations.
- **SE:** modular code, validation, error handling, tests, README, reproducible setup, cached model training, and session-based prediction/chat state.

## Important methodology notes
- For responsiveness, datasets above 50,000 rows are reproducibly sampled to 50,000 rows for model comparison and interactive model training. The UI still shows the full dataset overview.
- A stratified train/test split is used where class counts allow it.
- Imputation, scaling, and one-hot encoding are included in model pipelines to reduce preprocessing leakage.
- Macro-averaged precision, recall, and F1-score treat each disease class equally.
- The app reports comparative metrics; the top-scoring model is not necessarily clinically safe.
- The current app retrains models during a session rather than persisting them in a database. No database is required.

## Troubleshooting
- **Dataset not found:** place the CSV in the `data` folder with the exact filename above.
- **No target column detected:** rename the target column to `Disease` or update target detection in `src/data_utils.py`.
- **Training is slow:** a dataset with many disease classes or many rows can take time, especially for SVM. Start with the default settings and wait for training to finish.
- **PowerShell activation issue:** use the Command Prompt activation command shown above.
- **Python version issues:** use Python 3.11 or 3.12 and recreate the virtual environment.

## Current offline chatbot limitations
- Disease-specific reports are curated for selected common conditions. If a label is not covered, the app explains this limitation and gives safe next steps instead of inventing a disease-specific treatment.
- When Ollama is running, the chatbot uses a local language model for flexible questions. If Ollama is stopped or the model is missing, it falls back to keyword/rule matching, so specialized questions may not be understood. It does not send prompts to a paid external API.
- Medical descriptions and safety guidance should be reviewed by a qualified clinician before any real-world clinical use.

## Future improvements
- Expand the disease knowledge base using clinician-reviewed information and vetted references.
- Add model explainability (for example, feature importance for tree-based models) with clear limits on interpretation.
- Add automated tests for data parsing, prediction workflows, and more chatbot intents.
- Consider a database only if prediction history, users, or admin features are required.


## Ollama troubleshooting
- **`ollama` is not recognized:** close and reopen PowerShell after installing Ollama, then run `ollama --version`. If it still fails, repair/reinstall Ollama and ensure it is on PATH.
- **Connection refused / Ollama not detected:** start the Ollama app from the Windows Start menu, then test `http://localhost:11434/api/tags` in a browser.
- **Model not found:** run `ollama pull qwen2.5:3b` and check the exact name with `ollama list`.
- **First answer is slow:** the first model response can take longer while the model loads. Smaller models generally respond faster but may be less capable.
- **No GPU / limited RAM:** try a smaller model supported by Ollama, close other heavy apps, and allow several minutes for the first download.
- **Chatbot gives fallback answer:** make sure Ollama is running and that `OLLAMA_MODEL` matches a name shown by `ollama list`.

## Updated prediction workflow
- Open **Disease Prediction** and select one or more symptoms.
- Click **Predict possible disease**. Model training/evaluation and algorithm selection happen in the backend; the evaluation table and algorithm dropdown are hidden from the user-facing dashboard.
- The backend selects the successfully trained classifier with the highest macro F1-score on the held-out evaluation split, then uses it to predict from the selected symptoms.
- The selected algorithm is recorded with the prediction and shown with the result for transparency.
- After a prediction, use **Analyze This Disease** and the follow-up question field to ask about that predicted label.
- **Prediction History** is saved locally to `data/prediction_history.json`; individual entries can be removed, or the full history can be cleared from the History page.
- History is local to this project folder. Back up `data/prediction_history.json` if you want to preserve it across folder changes.


### First-run training speed

The app trains and evaluates six classifiers on a reproducible sample of up to 3,000 rows so the first run is more practical on a student laptop. The app caches the trained models for the current Streamlit session; later widget changes should not retrain them. Metrics therefore describe the sampled train/test split, not the entire dataset. If training still appears stuck, stop the server with `Ctrl+C`, close other Python processes, and restart with `python -m streamlit run app.py`.

## Persistent trained-model cache (Joblib)

On the first prediction, the app trains and evaluates the available algorithms and saves the fitted pipelines to `models/trained_models.joblib`. It also writes `models/model_metadata.json` with the dataset fingerprint, package version, evaluation metrics, and training details. On later runs, the app loads this saved artifact instead of repeating training when the dataset, feature configuration, training settings, and scikit-learn version match. If the dataset or compatible training configuration changes, or the saved cache is missing/corrupt, the app retrains and replaces the cache.

The model cache is generated locally by this application. Do not load Joblib files from unknown or untrusted sources because Joblib uses Python serialization.
