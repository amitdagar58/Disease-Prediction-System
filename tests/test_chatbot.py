from src.chatbot import answer_health_question
from src.disease_info import disease_report


def test_emergency_message():
    response = answer_health_question("I have difficulty breathing")
    assert "emergency" in response.lower()


def test_no_prescription():
    response = answer_health_question("What medicine dosage should I take?")
    assert "can't safely recommend" in response.lower()


def test_model_explanation():
    response = answer_health_question("How does the Random Forest model work?")
    assert "many decision trees" in response.lower()


def test_current_prediction_context():
    response = answer_health_question("How can I get relief from this condition?", current_disease="spondylosis")
    assert "general relief" in response.lower()
    assert "spondylosis" in response.lower()


def test_disease_analysis_unknown_label_is_honest():
    response = disease_report("unknown dataset label")
    assert "does not contain a verified disease-specific profile" in response.lower()


def test_no_api_key_required():
    response = answer_health_question("What is diabetes?")
    assert "diabetes" in response.lower()
    assert "openai" not in response.lower()
    assert "api key" not in response.lower()
