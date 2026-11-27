"""Offline educational disease information for the Disease Prediction System.

This is a curated rule-based knowledge base, not a clinical decision engine.
Unknown labels receive cautious, clearly-limited general guidance.
"""
from __future__ import annotations
import re

DISCLAIMER = (
    "\n\n**Safety note:** This information is for education only. A dataset prediction is not a diagnosis. "
    "A qualified healthcare professional can assess symptoms, medical history, and tests."
)

PROFILES = {
    "spondylosis": {
        "title": "Spondylosis (age-related changes in the spine)",
        "overview": "Spondylosis describes wear-and-tear changes in the spine. It may affect the neck (cervical spine) or lower back. Some people have changes on imaging without symptoms.",
        "symptoms": "Possible symptoms include neck or back pain, stiffness, reduced comfortable movement, or sometimes pain, tingling, numbness, or weakness travelling into an arm or leg.",
        "causes": "It is commonly associated with age-related changes in spinal discs and joints. Posture, repetitive loading, previous injury, and individual health factors may influence symptoms. The label alone cannot identify the cause of a person's pain.",
        "relief": ["Keep gently active within your comfort; short, easy walks may be useful.", "Take regular movement breaks from prolonged sitting and adjust screen height to a comfortable position.", "A warm or cold pack wrapped in a towel may ease discomfort; avoid direct skin contact and stop if irritation occurs.", "Avoid heavy lifting or movements that clearly worsen symptoms, but do not stay in bed for long periods unless a clinician advises it.", "A clinician or physiotherapist can advise whether specific exercises are suitable. Ask a clinician or pharmacist before using pain medicines, especially if you have other conditions or take other medicines."],
        "care": "Arrange a medical assessment for persistent, recurrent, worsening pain, pain after an injury, or pain with numbness/tingling. Seek urgent care for new or worsening limb weakness, difficulty walking or balancing, numbness around the groin/saddle area, or new loss of bladder or bowel control.",
        "prevention": "Regular appropriate activity, comfortable ergonomics, safe lifting, and breaks from one position may support spinal health, although age-related changes cannot always be prevented.",
    },
    "pain disorder affecting the neck": {
        "title": "Pain disorder affecting the neck (dataset label)",
        "overview": "This is a broad label describing a neck-pain-related problem. It does not tell us the exact cause, and neck pain can arise from muscles, joints, discs, nerves, injury, or other conditions.",
        "symptoms": "Neck discomfort, stiffness, or pain with movement may occur. Symptoms vary, and the model label cannot confirm which symptoms or cause apply to an individual.",
        "causes": "Possible causes include muscle strain, prolonged posture, injury, degenerative changes, or nerve irritation. A healthcare professional can assess the actual cause.",
        "relief": ["Take gentle movement breaks and avoid staying in one position for long periods.", "Keep screens near eye level and support the neck in a comfortable neutral position.", "Try a warm or cold pack wrapped in a towel for short periods if comfortable.", "Avoid forceful neck manipulation and activities that sharply worsen pain.", "A clinician or physiotherapist can recommend suitable exercises. Ask a clinician or pharmacist before taking medicines."],
        "care": "Seek assessment if pain persists, worsens, follows an injury, or comes with fever or other concerning symptoms. Seek urgent care for new limb weakness, trouble walking, severe sudden headache with neck stiffness, or loss of bladder/bowel control.",
        "prevention": "Take screen breaks, use comfortable ergonomics, build activity gradually, and use safe lifting techniques. These measures cannot prevent every cause of neck pain.",
    },
    "common cold": {"title":"Common cold", "overview":"A common cold is a viral infection of the upper respiratory tract and usually improves over time.", "symptoms":"Runny or blocked nose, sneezing, sore throat, cough, and sometimes mild fever or tiredness.", "causes":"Different respiratory viruses can cause colds; spread often occurs through respiratory droplets and contaminated hands/surfaces.", "relief":["Rest and drink fluids regularly.","Use saline nasal spray or drops if suitable and follow the product instructions.","Warm drinks may soothe a sore throat; avoid smoke and other irritants.","Ask a clinician or pharmacist before using over-the-counter products, especially for children, pregnancy, or existing conditions."], "care":"Seek medical advice for breathing difficulty, chest pain, dehydration, persistent high fever, symptoms that worsen, or symptoms that do not improve as expected.", "prevention":"Wash hands, improve ventilation, cover coughs and sneezes, and avoid close contact when unwell."},
    "asthma": {"title":"Asthma", "overview":"Asthma is a condition in which airways can become inflamed and narrowed, causing symptoms that may vary over time.", "symptoms":"Wheezing, coughing, chest tightness, and shortness of breath can occur, often with triggers or at night.", "causes":"Triggers differ and can include allergens, respiratory infections, smoke, exercise, or cold air. Only a clinician can assess whether symptoms are asthma.", "relief":["Follow the personal asthma action plan provided by your clinician.","Use prescribed inhalers exactly as directed; this chatbot cannot choose or change medicines.","Reduce known triggers where practical and keep follow-up appointments."], "care":"Severe breathing difficulty, blue/grey lips, confusion, or symptoms not improving with the person's prescribed emergency plan require emergency help.", "prevention":"Correct inhaler technique, an action plan, avoiding known triggers, and regular clinical review can help control asthma."},
    "diabetes": {"title":"Diabetes", "overview":"Diabetes is a group of conditions in which blood glucose is too high because the body does not make enough insulin or cannot use it effectively.", "symptoms":"Some people have increased thirst or urination, fatigue, blurred vision, or unexplained weight changes; type 2 diabetes may have few early symptoms.", "causes":"Type 1, type 2, and other forms have different causes and risk factors. Diagnosis requires appropriate medical testing.", "relief":["Follow the care plan made with your healthcare team.","Do not stop or change diabetes medicines based on chatbot advice.","Balanced meals, activity suitable for your health, monitoring as advised, and routine follow-up can support management."], "care":"Seek prompt help for severe weakness, confusion, vomiting with rapid/deep breathing, or severe symptoms of very low or very high blood glucose.", "prevention":"Some type 2 diabetes risk can be reduced through sustainable activity, balanced eating, and health checks, but not all diabetes is preventable."},
    "migraine": {"title":"Migraine", "overview":"Migraine is a neurological condition that can cause recurring headache attacks and other symptoms.", "symptoms":"Moderate or severe headache, nausea, sensitivity to light or sound, and sometimes visual or sensory aura.", "causes":"Triggers vary and may include disrupted sleep, stress, hormonal changes, missed meals, or particular sensory stimuli.", "relief":["Rest in a quiet, dim room if this helps.","Drink water and try to keep regular sleep and meal routines.","Keep a headache diary to identify patterns.","Ask a clinician or pharmacist about suitable treatment; avoid frequent use of pain medicines without advice."], "care":"Seek urgent care for a sudden, extremely severe headache, new weakness or speech/vision problems, confusion, fever with stiff neck, or headache after a significant injury.", "prevention":"Regular sleep, hydration, meals, and tracking personal triggers may reduce attacks for some people."},
    "hypertension": {"title":"High blood pressure (hypertension)", "overview":"Hypertension means blood pressure remains higher than recommended. It often causes no obvious symptoms and is assessed with blood-pressure measurements.", "symptoms":"Often none; a headache alone cannot confirm high blood pressure.", "causes":"Risk can be affected by genetics, age, kidney or hormonal conditions, some medicines, diet, activity, and other factors.", "relief":["Measure blood pressure with an appropriate device and discuss results with a healthcare professional.","Follow the prescribed plan and do not stop medicines without advice.","A balanced diet, regular suitable activity, limiting excess salt, and avoiding tobacco can support cardiovascular health."], "care":"Very high readings with chest pain, severe breathlessness, confusion, weakness, or vision/speech changes need emergency assessment.", "prevention":"Regular checks, healthy habits, and treatment when indicated can reduce complications."},
    "urinary tract infection": {"title":"Urinary tract infection (UTI)", "overview":"A UTI is an infection in part of the urinary system. Diagnosis may require clinical assessment and urine testing.", "symptoms":"Burning during urination, frequent or urgent urination, and lower abdominal discomfort may occur.", "causes":"Many UTIs are caused by bacteria entering the urinary tract. Similar symptoms can have other causes.", "relief":["Drink fluids as appropriate for your health unless a clinician has restricted them.","Arrange medical advice; a confirmed bacterial infection may need clinician-prescribed treatment.","Do not use leftover antibiotics or share medicines."], "care":"Seek prompt care for fever, chills, back/flank pain, vomiting, pregnancy, or feeling very unwell. Confusion or severe illness needs urgent help.", "prevention":"Normal hydration and avoiding delay when needing to urinate may help some people, but UTIs can still occur."},
    "pneumonia": {"title":"Pneumonia", "overview":"Pneumonia is an infection of the lungs that can be caused by different organisms and may range from mild to serious.", "symptoms":"Cough, fever, chest pain with breathing, tiredness, and shortness of breath may occur.", "causes":"Viruses, bacteria, and less commonly other organisms can cause pneumonia; assessment is needed to determine the cause and severity.", "relief":["Rest and drink fluids if you can do so safely.","Follow medical advice and take prescribed treatment as directed.","Avoid smoking and do not self-start antibiotics."], "care":"Breathing difficulty, blue/grey lips, confusion, severe chest pain, or rapidly worsening illness requires urgent/emergency care.", "prevention":"Vaccines recommended for your age/risk, hand hygiene, ventilation, and avoiding smoking can lower some risks."},
    "gastroesophageal reflux disease (gerd)": {"title":"Acid reflux / GERD", "overview":"Reflux occurs when stomach contents flow back into the oesophagus; frequent troublesome reflux may be called GERD.", "symptoms":"Heartburn, sour taste, or regurgitation can occur, but chest discomfort may have other causes.", "causes":"Contributors can include certain meals, lying down soon after eating, pregnancy, or other factors.", "relief":["Try smaller meals and avoid lying down for about 2–3 hours after eating if reflux occurs.","Notice and limit personal food or drink triggers.","If symptoms are frequent, discuss them with a healthcare professional before relying on long-term medicines."], "care":"Chest pressure/pain with breathlessness, sweating, faintness, or pain spreading to arm/jaw requires emergency assessment. Seek review for swallowing difficulty, vomiting blood, black stools, or unexplained weight loss.", "prevention":"Identifying triggers, avoiding late large meals, and discussing weight or medication factors with a clinician may help."},
    "depression": {"title":"Depression", "overview":"Depression is a mental health condition that can affect mood, interest, sleep, energy, concentration, and daily life. It is treatable.", "symptoms":"Persistent low mood or loss of interest, changes in sleep/appetite, low energy, guilt, or difficulty concentrating may occur.", "causes":"Biological, psychological, and social factors can contribute; a professional can assess symptoms and other possible causes.", "relief":["Reach out to a trusted person and consider speaking with a mental health professional.","Gentle routine, sleep, activity, and regular meals may support wellbeing but are not substitutes for treatment when needed.","Do not stop prescribed treatment without discussing it with your clinician."], "care":"If you may harm yourself or cannot stay safe, contact emergency services or a local crisis service now and stay with a trusted person if possible.", "prevention":"Supportive relationships, early help, and addressing stressors may help, but depression is not always preventable."},
    "anxiety": {"title":"Anxiety", "overview":"Anxiety can involve persistent worry or fear and physical symptoms. Occasional anxiety is common; persistent or disruptive symptoms deserve support.", "symptoms":"Restlessness, racing thoughts, muscle tension, sleep difficulty, palpitations, or stomach discomfort may occur.", "causes":"Stress, health factors, substances, and other conditions can contribute; assessment can help clarify the cause.", "relief":["Try slow, comfortable breathing without forcing deep breaths.","Break tasks into manageable steps and maintain a regular sleep routine.","Consider a mental health professional if symptoms persist or affect daily life."], "care":"Chest pain or breathing problems should not automatically be assumed to be anxiety. Seek urgent help for severe symptoms or if you cannot stay safe.", "prevention":"Stress management, adequate sleep, activity, and early support can help some people."},
    "acute bronchitis": {"title":"Acute bronchitis", "overview":"Acute bronchitis is inflammation of the large airways, commonly following a viral infection; cough can last for several weeks.", "symptoms":"Cough, sometimes mucus, fatigue, mild fever, or chest discomfort may occur.", "causes":"Viruses are common causes; smoke and irritants can worsen airway symptoms.", "relief":["Rest and drink fluids.","Avoid smoking, vaping, and second-hand smoke.","Ask a clinician or pharmacist before using cough medicines; antibiotics do not help viral infections."], "care":"Seek urgent care for difficulty breathing, blue/grey lips, coughing blood, chest pain, or severe/worsening illness; get advice for persistent fever or high-risk health conditions.", "prevention":"Hand hygiene, respiratory etiquette, ventilation, and avoiding smoke can reduce risk or irritation."},
    "kidney stone": {"title":"Kidney stone", "overview":"A kidney stone is a hard deposit that forms from substances in urine. Size and location affect symptoms and treatment.", "symptoms":"Severe side/back pain that may come in waves, nausea, vomiting, or blood in urine can occur.", "causes":"Stone formation is influenced by urine concentration, mineral composition, genetics, diet, and some medical conditions.", "relief":["Seek medical advice for suspected stones; treatment depends on stone size and location.","Drink fluids if you can and have not been told to restrict them, but do not delay care for severe symptoms.","Do not self-treat with unverified remedies."], "care":"Urgent assessment is needed for fever/chills with flank pain, inability to urinate, uncontrolled pain, persistent vomiting, or feeling very unwell.", "prevention":"For some people, adequate fluids and tailored dietary changes help; prevention advice depends on stone analysis and medical history."},
    "gout": {"title":"Gout", "overview":"Gout is an inflammatory arthritis caused by urate crystals in a joint.", "symptoms":"A sudden painful, hot, swollen joint—often the big toe—may occur.", "causes":"High urate levels can contribute, influenced by genetics, kidney function, medicines, and other factors.", "relief":["Rest and protect the painful joint while arranging appropriate medical advice.","A wrapped cold pack may help discomfort for short periods.","Ask a clinician about treatment; do not start or stop medicines based on this chatbot."], "care":"A first hot, swollen joint or joint symptoms with fever needs prompt assessment because infection can look similar.", "prevention":"Long-term prevention may involve clinician-guided urate management and individual lifestyle advice."},
    "arthritis": {"title":"Arthritis (general term)", "overview":"Arthritis refers to several conditions that cause joint pain, swelling, or stiffness; the type matters for treatment.", "symptoms":"Joint pain, swelling, stiffness, and reduced movement can occur.", "causes":"Causes differ and include wear-related, autoimmune, crystal-related, and infectious conditions.", "relief":["Use gentle activity within comfort and avoid sudden overloading.","Heat or cold wrapped in a towel may help some people.","A clinician can identify the type and suggest suitable exercise or treatment."], "care":"A suddenly hot, swollen joint, especially with fever, requires urgent assessment.", "prevention":"Appropriate activity, injury prevention, and condition-specific management can help function."},
    "asthma": {"title":"Asthma", "overview":"Asthma is a condition in which airways can become inflamed and narrowed, causing symptoms that may vary over time.", "symptoms":"Wheezing, coughing, chest tightness, and shortness of breath can occur, often with triggers or at night.", "causes":"Triggers differ and can include allergens, respiratory infections, smoke, exercise, or cold air. Only a clinician can assess whether symptoms are asthma.", "relief":["Follow the personal asthma action plan provided by your clinician.","Use prescribed inhalers exactly as directed; this chatbot cannot choose or change medicines.","Reduce known triggers where practical and keep follow-up appointments."], "care":"Severe breathing difficulty, blue/grey lips, confusion, or symptoms not improving with the person's prescribed emergency plan require emergency help.", "prevention":"Correct inhaler technique, an action plan, avoiding known triggers, and regular clinical review can help control asthma."},
}

ALIASES = {
    "neck pain": "pain disorder affecting the neck", "neck pain disorder": "pain disorder affecting the neck",
    "pain disorder affecting the neck": "pain disorder affecting the neck", "spondylosis": "spondylosis",
    "cold": "common cold", "flu": "common cold", "high blood pressure": "hypertension",
    "blood pressure": "hypertension", "uti": "urinary tract infection", "urine infection": "urinary tract infection",
    "acid reflux": "gastroesophageal reflux disease (gerd)", "gerd": "gastroesophageal reflux disease (gerd)",
    "headache": "migraine", "joint pain": "arthritis", "joint swelling": "arthritis",
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def find_profile(disease: str | None):
    if not disease:
        return None
    raw = str(disease).strip()
    norm = normalize(raw)
    key = ALIASES.get(norm, raw.lower())
    if key in PROFILES:
        return PROFILES[key]
    for name, profile in PROFILES.items():
        if normalize(name) == norm or normalize(name) in norm or norm in normalize(name):
            return profile
    return None


def disease_report(disease: str) -> str:
    profile = find_profile(disease)
    if not profile:
        return (
            f"## About: {disease}\n\n"
            "This is the label returned by your dataset/model. The label alone does not reliably explain the exact medical condition, cause, severity, or treatment. "
            "The local knowledge base does not contain a verified disease-specific profile for this label, so it would be unsafe to invent one.\n\n"
            "### Safe next steps\n"
            "- Write down your symptoms, when they started, what makes them better/worse, and any medicines you take.\n"
            "- Discuss the symptoms and this model output with a qualified healthcare professional.\n"
            "- Do not start, stop, or change medicine based on this prediction.\n"
            "- Seek urgent care for severe, sudden, or rapidly worsening symptoms, breathing difficulty, chest pain, new weakness, confusion, or loss of bladder/bowel control."
            + DISCLAIMER
        )
    relief = "\n".join(f"- {item}" for item in profile["relief"])
    return (
        f"## {profile['title']}\n\n"
        f"### What it means\n{profile['overview']}\n\n"
        f"### Possible symptoms\n{profile['symptoms']}\n\n"
        f"### Possible causes and contributing factors\n{profile['causes']}\n\n"
        f"### General relief and self-care\n{relief}\n\n"
        f"### When to seek medical care\n{profile['care']}\n\n"
        f"### Prevention and risk reduction\n{profile['prevention']}"
        + DISCLAIMER
    )


def get_known_disease_names() -> list[str]:
    return sorted(PROFILES.keys())
