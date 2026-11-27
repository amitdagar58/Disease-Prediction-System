import pandas as pd
from pathlib import Path

def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put Final_Augmented_dataset_Diseases_and_Symptoms.csv in the data/ folder."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The CSV dataset is empty.")
    df.columns = [str(c).strip() for c in df.columns]

    # Detect the disease/label column by common names, then fall back to the last column.
    target_candidates = [c for c in df.columns if any(k in c.lower() for k in ("disease", "prognosis", "diagnosis", "target", "label"))]
    target_col = target_candidates[0] if target_candidates else df.columns[-1]

    # Remove rows without a target label and normalize target labels to strings.
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[df[target_col] != ""].reset_index(drop=True)

    # Identify symptom feature columns. If one column contains a symptom list, expand it to binary columns.
    symptom_list_cols = [
        c for c in df.columns if c != target_col and any(k in c.lower() for k in ("symptom", "symptoms"))
    ]
    data_mode = "one symptom per column"
    if len(symptom_list_cols) == 1 and df[symptom_list_cols[0]].astype(str).str.contains(r"[,;|]").any():
        col = symptom_list_cols[0]
        rows = df[col].fillna("").astype(str).apply(
            lambda s: [x.strip().lower().replace(" ", "_") for x in __import__("re").split(r"[,;|]", s) if x.strip()]
        )
        vocabulary = sorted({symptom for row in rows for symptom in row})
        expanded = pd.DataFrame(0, index=df.index, columns=vocabulary, dtype=int)
        for idx, symptoms in rows.items():
            for symptom in symptoms:
                expanded.loc[idx, symptom] = 1
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
        data_mode = "symptom list expanded into binary features"

    feature_cols = [c for c in df.columns if c != target_col]
    # Drop identifiers and obvious non-predictive columns.
    feature_cols = [c for c in feature_cols if not any(k in c.lower() for k in ("unnamed:", "patient_id", "record_id", "index"))]
    if not feature_cols:
        raise ValueError("No symptom/features found after excluding the target column.")

    # Standardize common binary symptom strings to 0/1; leave other categorical values for preprocessing.
    for c in feature_cols:
        if df[c].dtype == object:
            values = set(df[c].dropna().astype(str).str.strip().str.lower().unique())
            mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "present": 1, "absent": 0}
            if values and values.issubset(mapping):
                df[c] = df[c].astype(str).str.strip().str.lower().map(mapping)
    return df, target_col, feature_cols, data_mode

def inspect_dataset(df, target_col):
    summary = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Disease classes", "Missing cells", "Duplicate rows"],
        "Value": [len(df), df.shape[1], df[target_col].nunique(), int(df.isna().sum().sum()), int(df.duplicated().sum())]
    })
    return summary
