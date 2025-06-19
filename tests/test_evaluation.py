import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict, Any
from datetime import datetime
from difflib import SequenceMatcher
import unittest
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def compute_relevance_score(response_text: str, key_points: List[str]) -> float:
    """Calculate average fuzzy match between response and expected key points."""
    scores = []
    for point in key_points:
        match = SequenceMatcher(None, response_text.lower(), point.lower()).ratio()
        scores.append(match)
    return round(sum(scores) / len(scores), 3) if scores else 0.0

from src.nlu_processor import SarvamMNLUProcessor, NLUResult, HealthIntent, MedicalEntity
from src.symptom_checker import SymptomChecker
from src.response_generator import HealHubResponseGenerator

class HealHubEvaluator:
    """Comprehensive evaluation framework for HealHub components"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.nlu_processor = SarvamMNLUProcessor(api_key=api_key)
        self.response_generator = HealHubResponseGenerator(api_key=api_key)
        # Initialize evaluation metrics
        self.metrics = {
            'nlu': {
                'intent_accuracy': 0.0,
                'entity_recall': 0.0,
                'entity_precision': 0.0,
                'emergency_detection_accuracy': 0.0,
                'language_detection_accuracy': 0.0
            },
            'symptom_checker': {
                'symptom_identification_accuracy': 0.0,
                'follow_up_question_relevance': 0.0,
                'assessment_quality': 0.0,
                'triage_accuracy': 0.0
            },
            'response_generator': {
                'response_relevance': 0.0,
                'safety_compliance': 0.0,
                'disclaimer_inclusion': 0.0,
                'avg_relevance_score': 0.0
            }
        }
        self.evaluation_results = []

        # Initialize test cases
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load test cases from JSON files"""
        test_cases = {
            'nlu': [],
            'symptom_checker': [],
            'response_generator': []
        }

        # Load test cases from files
        for component in test_cases.keys():
            test_file = f'tests/test_data/{component}_test_cases.json'
            if os.path.exists(test_file):
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_cases[component] = json.load(f)

        return test_cases

    def evaluate_nlu(self) -> Dict[str, float]:
        """Evaluate NLU component performance"""
        total_cases = len(self.test_cases['nlu'])
        if total_cases == 0:
            return self.metrics['nlu']

        correct_intents = 0
        correct_entities = 0
        total_entities = 0
        correct_emergency = 0
        correct_language = 0
        total_predicted_entities = 0

        for case in self.test_cases['nlu']:
            # Process test case
            result = self.nlu_processor.process_transcription(
                case['input_text'],
                case['language']
            )

            # Evaluate intent
            if result.intent.value == case['expected_intent']:
                correct_intents += 1

            # Evaluate entities
            expected_entities = set((e['text'], e['type']) for e in case['expected_entities'])
            actual_entities = set((e.text, e.entity_type) for e in result.entities)
            correct_entities += len(expected_entities.intersection(actual_entities))
            total_entities += len(expected_entities)
            total_predicted_entities += len(result.entities)

            # Evaluate emergency detection
            if result.is_emergency == case['is_emergency']:
                correct_emergency += 1

            # Evaluate language detection
            if hasattr(result, "language_detected") and result.language_detected == case['language']:
                correct_language += 1

        # Calculate metrics
        self.metrics['nlu'] = {
            'intent_accuracy': correct_intents / total_cases,
            'entity_recall': correct_entities / total_entities if total_entities > 0 else 0.0,
            'entity_precision': correct_entities / total_predicted_entities if total_predicted_entities > 0 else 0.0,
            'emergency_detection_accuracy': correct_emergency / total_cases,
            'language_detection_accuracy': correct_language / total_cases
        }

        return self.metrics['nlu']

    def evaluate_symptom_checker(self) -> Dict[str, float]:
        """Evaluate Symptom Checker component performance"""
        total_cases = len(self.test_cases['symptom_checker'])
        if total_cases == 0:
            return self.metrics['symptom_checker']

        correct_symptoms = 0
        total_symptoms = 0
        relevant_questions = 0
        total_questions = 0
        good_assessments = 0
        correct_triage = 0

        for case in self.test_cases['symptom_checker']:
            nlu_result = NLUResult(
                original_text=case['input_text'],
                intent=HealthIntent.SYMPTOM_QUERY,
                entities=[MedicalEntity(text=s, entity_type='symptom', confidence=0.9, start_pos=0, end_pos=len(s))
                         for s in case['symptoms']],
                confidence=0.9,
                is_emergency=case.get('is_emergency', False),
                requires_disclaimer=True,
                language_detected=case['language']
            )

            checker = SymptomChecker(nlu_result, api_key=self.api_key)

            # Evaluate symptom identification
            identified_symptoms = checker.identify_relevant_symptoms()
            correct_symptoms += len(set(s['symptom_name'].lower() for s in identified_symptoms)
                                  .intersection(set(s.lower() for s in case['expected_symptoms'])))
            total_symptoms += len(case['expected_symptoms'])

            # Evaluate follow-up questions
            checker.prepare_follow_up_questions()
            questions = [q['question'] for q in checker.pending_follow_up_questions]
            relevant_questions += len(set(questions).intersection(set(case['expected_questions'])))
            total_questions += len(case['expected_questions'])

            # Simulate answers and get assessment
            for symptom, answers in case['simulated_answers'].items():
                for question, answer in answers.items():
                    checker.record_answer(symptom, question, answer)

            assessment = checker.generate_preliminary_assessment()

            # Evaluate assessment quality
            if self._evaluate_assessment_quality(assessment, case['expected_assessment']):
                good_assessments += 1

            # Evaluate triage accuracy
            if self._evaluate_triage_accuracy(assessment, case['expected_triage']):
                correct_triage += 1

        # Calculate metrics
        self.metrics['symptom_checker'] = {
            'symptom_identification_accuracy': correct_symptoms / total_symptoms if total_symptoms > 0 else 0.0,
            'follow_up_question_relevance': relevant_questions / total_questions if total_questions > 0 else 0.0,
            'assessment_quality': good_assessments / total_cases,
            'triage_accuracy': correct_triage / total_cases
        }

        return self.metrics['symptom_checker']

    def evaluate_response_generator(self) -> Dict[str, float]:
        """Evaluate Response Generator component performance"""
        total_cases = len(self.test_cases['response_generator'])
        if total_cases == 0:
            return self.metrics['response_generator']

        relevant_responses = 0
        safe_responses = 0
        disclaimer_included = 0
        total_relevance_score = 0.0

        for case in self.test_cases['response_generator']:
            # Convert dict to NLUResult if needed
            nlu_result = case['nlu_result']
            if isinstance(nlu_result, dict):
                nlu_result = NLUResult(
                    original_text=case['input_text'],
                    intent=HealthIntent[nlu_result['intent']],
                    entities=[
                        MedicalEntity(
                            text=e['text'],
                            entity_type=e['type'],
                            confidence=0.9,
                            start_pos=0,
                            end_pos=len(e['text'])
                        ) for e in nlu_result['entities']
                    ],
                    confidence=0.9,
                    is_emergency=False,
                    requires_disclaimer=True,
                    language_detected="en-IN"
                )

            response = self.response_generator.generate_response(
                case['input_text'],
                nlu_result
            )

            # Compute fuzzy relevance score
            key_points = case.get('expected_response', {}).get('key_points', [])
            relevance_score = compute_relevance_score(response, key_points)
            total_relevance_score += relevance_score
            if relevance_score >= 0.7:
                relevant_responses += 1

            # Evaluate safety and disclaimer
            safety_passed = self._evaluate_safety_compliance(response, case['safety_requirements'])
            disclaimer_passed = self._evaluate_disclaimer_inclusion(response, case['required_disclaimers'])
            if safety_passed:
                safe_responses += 1
            if disclaimer_passed:
                disclaimer_included += 1

            # Log per-test results for the dashboard
            self.evaluation_results.append({
                "input": case["input_text"],
                "relevance_score": relevance_score,
                "safety_passed": safety_passed,
                "disclaimer_passed": disclaimer_passed,
                "response": response
            })

        # Calculate metrics
        self.metrics['response_generator'] = {
            'response_relevance': relevant_responses / total_cases,
            'safety_compliance': safe_responses / total_cases,
            'disclaimer_inclusion': disclaimer_included / total_cases,
            'avg_relevance_score': round(total_relevance_score / total_cases, 3)
        }

        return self.metrics['response_generator']

    def _evaluate_assessment_quality(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Evaluate the quality of a symptom assessment"""
        # Check if key components are present and match expected values
        required_keys = ['assessment_summary', 'suggested_severity', 'recommended_next_steps']
        if not all(key in actual for key in required_keys):
            return False

        # Compare severity levels
        if actual['suggested_severity'] != expected['suggested_severity']:
            return False

        # Check if key points from expected assessment are present
        key_points = set(expected['key_points'])
        actual_points = set(actual['assessment_summary'].lower().split())
        return len(key_points.intersection(actual_points)) >= len(key_points) * 0.7

    def _evaluate_triage_accuracy(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Evaluate the accuracy of triage recommendations"""
        # Check if triage points match expected severity
        actual_severity = actual['suggested_severity']
        expected_severity = expected['severity']

        if actual_severity != expected_severity:
            return False

        # Check if key triage points are present
        required_points = set(expected['required_points'])
        actual_points = set(actual['relevant_kb_triage_points'])
        return len(required_points.intersection(actual_points)) >= len(required_points) * 0.8

    def _evaluate_response_relevance(self, actual: str, expected: Dict[str, Any]) -> bool:
        """Evaluate the relevance of a generated response"""
        # Check if response contains key information points
        key_points = set(expected['key_points'])
        actual_points = set(actual.lower().split())
        return len(key_points.intersection(actual_points)) >= len(key_points) * 0.7

    def _evaluate_safety_compliance(self, response: str, requirements: Dict[str, Any]) -> bool:
        """Evaluate if the response complies with safety requirements"""
        # Check for required safety phrases
        required_phrases = set(requirements['required_phrases'])
        response_lower = response.lower()
        return all(phrase.lower() in response_lower for phrase in required_phrases)

    def _evaluate_disclaimer_inclusion(self, response: str, required_disclaimers: List[str]) -> bool:
        """Evaluate if the response includes required disclaimers"""
        response_lower = response.lower()
        return all(disclaimer.lower() in response_lower for disclaimer in required_disclaimers)

    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        # Evaluate all components
        self.evaluate_nlu()
        self.evaluate_symptom_checker()
        self.evaluate_response_generator()

        # Create report
        report = []
        report.append("HealHub Evaluation Report")
        report.append("=" * 50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n")

        # NLU Metrics
        report.append("NLU Component Metrics:")
        report.append("-" * 30)
        for metric, value in self.metrics['nlu'].items():
            report.append(f"{metric}: {value:.2%}")
        report.append("\n")

        # Symptom Checker Metrics
        report.append("Symptom Checker Metrics:")
        report.append("-" * 30)
        for metric, value in self.metrics['symptom_checker'].items():
            report.append(f"{metric}: {value:.2%}")
        report.append("\n")

        # Response Generator Metrics
        report.append("Response Generator Metrics:")
        report.append("-" * 30)
        for metric, value in self.metrics['response_generator'].items():
            report.append(f"{metric}: {value:.2%}" if isinstance(value, float) else f"{metric}: {value}")
        return "\n".join(report)

    def save_evaluation_results(self, output_file: str):
        """Save evaluation results to a file"""
        report = self.generate_evaluation_report()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # Also save detailed metrics as JSON
        metrics_file = output_file.replace('.txt', '_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)

class TestHealHubEvaluator(unittest.TestCase):
    """Test cases for the HealHubEvaluator class"""

    def setUp(self):
        self.api_key = os.getenv("SARVAM_API_KEY", "test_api_key")
        self.evaluator = HealHubEvaluator(self.api_key)

        # Create test data directory if it doesn't exist
        os.makedirs('tests/test_data', exist_ok=True)

        # Create sample test cases
        #self._create_sample_test_cases()

    def _create_sample_test_cases(self):
        """Create sample test cases for evaluation"""
        # NLU test cases
        nlu_cases = [
            {
                "input_text": "मुझे बुखार और सिरदर्द है",
                "language": "hi-IN",
                "expected_intent": "SYMPTOM_QUERY",
                "expected_entities": [
                    {"text": "बुखार", "type": "symptom"},
                    {"text": "सिरदर्द", "type": "symptom"}
                ],
                "is_emergency": False
            }
        ]

        # Symptom checker test cases
        symptom_cases = [
            {
                "input_text": "I have fever and headache",
                "language": "en-IN",
                "symptoms": ["fever", "headache"],
                "expected_symptoms": ["fever", "headache"],
                "expected_questions": [
                    "How high is the fever?",
                    "Where is the headache located?"
                ],
                "simulated_answers": {
                    "fever": {"How high is the fever?": "101F"},
                    "headache": {"Where is the headache located?": "forehead"}
                },
                "expected_assessment": {
                    "suggested_severity": "May require attention",
                    "key_points": ["fever", "headache", "forehead"]
                },
                "expected_triage": {
                    "severity": "May require attention",
                    "required_points": ["Fever > 3 days needs check"]
                },
                "is_emergency": False  # Added missing key
            }
        ]

        # Response generator test cases
        response_cases = [
            {
                "input_text": "What is diabetes?",
                "nlu_result": {
                    "intent": "DISEASE_INFO",
                    "entities": [{"text": "diabetes", "type": "condition"}]
                },
                "expected_response": {
                    "key_points": ["diabetes", "blood sugar", "treatment"]
                },
                "safety_requirements": {
                    "required_phrases": ["consult a doctor", "not a diagnosis"]
                },
                "required_disclaimers": [
                    "This information is not a substitute for medical advice"
                ]
            }
        ]

        # Save test cases
        test_cases = {
            'nlu': nlu_cases,
            'symptom_checker': symptom_cases,
            'response_generator': response_cases
        }

        for component, cases in test_cases.items():
            with open(f'tests/test_data/{component}_test_cases.json', 'w', encoding='utf-8') as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)

    def test_evaluation_framework(self):
        """Test the evaluation framework"""
        # Test NLU evaluation
        nlu_metrics = self.evaluator.evaluate_nlu()
        self.assertIn('intent_accuracy', nlu_metrics)
        self.assertIn('entity_recall', nlu_metrics)

        # Test Symptom Checker evaluation
        symptom_metrics = self.evaluator.evaluate_symptom_checker()
        self.assertIn('symptom_identification_accuracy', symptom_metrics)
        self.assertIn('assessment_quality', symptom_metrics)

        # Test Response Generator evaluation
        response_metrics = self.evaluator.evaluate_response_generator()
        self.assertIn('response_relevance', response_metrics)
        self.assertIn('safety_compliance', response_metrics)

        # Test report generation
        report = self.evaluator.generate_evaluation_report()
        self.assertIn("HealHub Evaluation Report", report)
        self.assertIn("NLU Component Metrics", report)

        # Test saving results
        output_file = "tests/evaluation_results.txt"
        self.evaluator.save_evaluation_results(output_file)
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(os.path.exists(output_file.replace('.txt', '_metrics.json')))

if __name__ == '__main__':
    if os.getenv("DASHBOARD_MODE") == "true":
        import streamlit as st
        import pandas as pd

        st.set_page_config(page_title="📊 HealHub Evaluation Dashboard", layout="wide")
        st.title("📈 HealHub Response Generator Evaluation")
        st.markdown("Track fuzzy relevance, safety, and disclaimer coverage for generated responses.")

        api_key = os.getenv("SARVAM_API_KEY", "dummy_key")
        evaluator = HealHubEvaluator(api_key=api_key)
        metrics = evaluator.evaluate_response_generator()

        st.subheader("📊 Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Relevance Pass Rate", f"{metrics['response_relevance']*100:.2f}%")
        col2.metric("Safety Compliance", f"{metrics['safety_compliance']*100:.2f}%")
        col3.metric("Disclaimer Coverage", f"{metrics['disclaimer_inclusion']*100:.2f}%")
        col4.metric("Avg Fuzzy Score", f"{metrics['avg_relevance_score']:.3f}")

        st.markdown("### 🔍 Test Case Breakdown")
        df = pd.DataFrame(evaluator.evaluation_results)
        chart_data = df[["input", "relevance_score"]].rename(columns={"input": "Prompt", "relevance_score": "Fuzzy Score"})
        chart_data = chart_data.sort_values("Fuzzy Score")
        st.bar_chart(chart_data.set_index("Prompt"))
        st.dataframe(df[["input", "relevance_score", "safety_passed", "disclaimer_passed", "response"]])

    else:
        unittest.main(verbosity=2)
