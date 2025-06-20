import json
import time
import re
import os
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from fuzzywuzzy import process
import textdistance
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("src/hinglish_symptoms.json", "r", encoding="utf-8") as f:
    HINGLISH_SYMPTOMS = json.load(f)

import re

def normalize_hinglish_terms(text: str) -> str:
    for key, variations in HINGLISH_SYMPTOMS.items():
        for variant in variations:
            pattern = r'(?<!\w)' + re.escape(variant) + r'(?!\w)'
            text = re.sub(pattern, key, text)
    return text

def tokenize_hinglish_query(text: str) -> List[str]:
    """Break down Hinglish sentence into individual tokens."""
    return re.findall(r'\b[\w]+\b', text.lower())

def load_common_misspellings(filepath="src/common_misspellings.json"):
    import json
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load misspellings dictionary: {e}")
        return {}

COMMON_MISSPELLINGS = load_common_misspellings()

@lru_cache(maxsize=256)
def correct_misspelled_entity(word: str) -> str:
    # Direct match to key
    if word.lower() in COMMON_MISSPELLINGS:
        return word.lower()
    # Match to value
    for canonical, variants in COMMON_MISSPELLINGS.items():
        if word.lower() in [v.lower() for v in variants]:
            return canonical
    # Fuzzy match to keys
    result = process.extractOne(word.lower(), COMMON_MISSPELLINGS.keys())
    if result is None:
        return word
    best_match, score = result
    return best_match if score > 80 else word

def phonetic_match(word: str, candidates_tuple: tuple) -> str:
    candidates = list(candidates_tuple)
    # Add all variant values to candidates
    for variants in COMMON_MISSPELLINGS.values():
        candidates.extend(variants)
    if not candidates:
        return word
    best_match = max(
        candidates,
        key=lambda c: textdistance.levenshtein.normalized_similarity(word.lower(), c.lower())
    )
    if textdistance.levenshtein.normalized_similarity(word.lower(), best_match.lower()) > 0.7:
        # If best_match is a variant, return its canonical key
        for canonical, variants in COMMON_MISSPELLINGS.items():
            if best_match.lower() == canonical.lower() or best_match.lower() in [v.lower() for v in variants]:
                return canonical
    return word

class HealthIntent(Enum):
    """Healthcare-specific intents"""
    SYMPTOM_QUERY = "symptom_query"
    DISEASE_INFO = "disease_info"
    MEDICATION_INFO = "medication_info"
    WELLNESS_TIP = "wellness_tip"
    EMERGENCY = "emergency"
    DIAGNOSIS_REQUEST = "diagnosis_request"
    PREVENTION_INFO = "prevention_info"
    GENERAL_HEALTH = "general_health"
    UNKNOWN = "unknown"

@dataclass
class MedicalEntity:
    """Represents extracted medical entities"""
    text: str
    entity_type: str  # symptom, disease, medication, body_part, etc.
    confidence: float
    start_pos: int
    end_pos: int

@dataclass
class NLUResult:
    """NLU processing result"""
    original_text: str
    intent: HealthIntent
    confidence: float
    entities: List[MedicalEntity]
    is_emergency: bool
    requires_disclaimer: bool
    language_detected: str

class SarvamAPIClient:
    """Client for Sarvam AI API services"""

    def __init__(self, api_key: Optional[str] = None):
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable or api_key parameter is required")

        self.base_url = "https://api.sarvam.ai"

    def chat_completion(self, messages: List[Dict], model: str = "sarvam-m", **kwargs) -> Dict:
        """
        Generate chat completion using Sarvam-M model

        Args:
            messages: List of message objects with role and content
            model: Model name (default: sarvam-m)
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        """
        url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "n": kwargs.get("n", 1)
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ Sarvam API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return {}

class SarvamMNLUProcessor:
    """NLU processor using Sarvam-M for healthcare queries"""

    def __init__(self, api_key: Optional[str] = None):
        self.sarvam_client = SarvamAPIClient(api_key)
        self.symptom_kb = None  # For storing symptom knowledge base
        self.emergency_keywords = {} # Initialize as empty dict
        self._load_keyword_config() # Load keywords from config file

        # Diagnosis request patterns
        self.diagnosis_patterns = [
            r'\b(what.*wrong|diagnose|what.*disease|what.*illness)\b',
            r'\b(do i have|am i suffering)\b',
            r'\b(क्या.*बीमारी|निदान)\b',  # Hindi
            r'\b(என்ன.*நோய்|கண்டறிதல்)\b',  # Tamil
        ]
        self._load_symptom_kb() # Load symptom knowledge base

    def _load_keyword_config(self, config_filepath="src/nlu_config.json"):
        """Loads keyword configurations from a JSON file."""
        try:
            with open(config_filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                keyword_lists = config_data.get("keyword_lists", {})
                self.emergency_keywords = keyword_lists.get("emergency_keywords", {})
                if self.emergency_keywords:
                    print(f"✅ Emergency keywords loaded successfully from {config_filepath}.")
                else:
                    print(f"⚠️ No emergency keywords found in {config_filepath} or structure is incorrect.")
                    self.emergency_keywords = {} # Ensure it's an empty dict if not found
        except FileNotFoundError:
            print(f"⚠️ Keyword config file not found at {config_filepath}. Emergency keyword detection will be limited.")
            self.emergency_keywords = {}
        except json.JSONDecodeError:
            print(f"⚠️ Error decoding JSON from keyword config file at {config_filepath}. Emergency keyword detection will be limited.")
            self.emergency_keywords = {}
        except Exception as e: # Catch any other unexpected errors during loading
            print(f"❌ An unexpected error occurred while loading keyword config from {config_filepath}: {e}")
            self.emergency_keywords = {}

    def _load_symptom_kb(self, filepath="src/symptom_knowledge_base.json"):
        """Loads the symptom knowledge base from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.symptom_kb = data.get("symptoms", [])
                print(f"✅ Symptom knowledge base loaded successfully from {filepath}.")
        except FileNotFoundError:
            print(f"⚠️ Symptom knowledge base file not found at {filepath}. Keyword matching will be limited.")
            self.symptom_kb = []
        except json.JSONDecodeError:
            print(f"⚠️ Error decoding JSON from symptom knowledge base file at {filepath}. Keyword matching will be limited.")
            self.symptom_kb = []

    def process_transcription(self, transcribed_text: str, source_language: str = "hi-IN") -> NLUResult:
        transcribed_text = normalize_hinglish_terms(transcribed_text)

            # Hinglish intent pre-check
        hinglish_intent = self.get_intent(transcribed_text)
        if hinglish_intent == HealthIntent.SYMPTOM_QUERY:
            intent, intent_confidence = hinglish_intent, 1.0
        else:
            intent, intent_confidence = self._classify_intent(transcribed_text, source_language)

        print(f"🧠 Processing NLU for: '{transcribed_text}'")

        """
        Process transcribed text through Sarvam-M for NLU

        Args:
            transcribed_text: Text from Saarika v2 STT
            source_language: Source language code

        Returns:
            NLUResult with intent, entities, and safety flags
        """
        print(f"🧠 Processing NLU for: '{transcribed_text}'")

        # Step 1: Safety checks first
        is_emergency = self._detect_emergency(transcribed_text, source_language)
        requires_disclaimer = self._requires_medical_disclaimer(transcribed_text)

        # Step 2: Intent classification using Sarvam-M
        intent, intent_confidence = self._classify_intent(transcribed_text, source_language)

        # Step 3: Entity extraction
        entities = self._extract_medical_entities(transcribed_text, source_language)

        # Step 4: Language detection refinement
        detected_language = self._detect_language(transcribed_text)

        result = NLUResult(
            original_text=transcribed_text,
            intent=intent,
            confidence=intent_confidence,
            entities=entities,
            is_emergency=is_emergency,
            requires_disclaimer=requires_disclaimer,
            language_detected=detected_language
        )

        print(f"✅ NLU Result - Intent: {intent.value}, Confidence: {intent_confidence:.2%}")
        print(f"🚨 Emergency: {is_emergency}, Disclaimer: {requires_disclaimer}")

        return result

    def _detect_emergency(self, text: str, language: str) -> bool:
        """Detect emergency situations"""
        text_lower = text.lower()

        lang_code = language.split('-')[0] if '-' in language else language

        # Safely get default English keywords
        default_emergency_words = self.emergency_keywords.get('en', [])
        # Get language-specific keywords, falling back to default (English) ones
        emergency_words = self.emergency_keywords.get(lang_code, default_emergency_words)

        # If no keywords are available (neither for the specific language nor for English fallback),
        # then no detection can occur.
        if not emergency_words:
            return False

        for keyword in emergency_words:
            if keyword.lower() in text_lower: # Ensure keyword itself is also lowercased for comparison
                return True

        return False

    def _requires_medical_disclaimer(self, text: str) -> bool:
        """Check if query requires medical disclaimer"""
        # Any health-related query should have disclaimer
        # This is a conservative approach for healthcare
        return True

    def _classify_intent(self, text: str, language: str) -> Tuple[HealthIntent, float]:
        """Classify intent using real Sarvam-M API"""

        messages = [
            {
                "role": "system",
                "content": """You are a healthcare intent classifier. Classify user queries into these categories:

1. symptom_query - User is describing one or more physical symptoms, feelings of illness, or specific pains (e.g., 'I have a headache and fever', 'my throat hurts').
2. disease_info - Information about diseases/conditions
3. medication_info - Medicine-related queries
4. wellness_tip - Health and wellness advice
5. emergency - Urgent medical situations
6. diagnosis_request - Seeking medical diagnosis
7. prevention_info - Disease prevention information
8. general_health - General health questions

Respond ONLY with JSON format: {"intent": "category_name", "confidence": 0.95}"""
            },
            {
                "role": "user",
                "content": f"Classify this healthcare query: '{text}'\nLanguage: {language}"
            }
        ]

        try:
            print(f"🔄 Calling Sarvam-M for intent classification...")
            response = self.sarvam_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )

            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                # Clean the response to extract JSON
                content = content.strip()
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].strip()

                result = json.loads(content)

                intent_str = result.get("intent", "unknown")
                confidence = result.get("confidence", 0.5)

                # Map to enum
                intent_mapping = {
                    "symptom_query": HealthIntent.SYMPTOM_QUERY,
                    "disease_info": HealthIntent.DISEASE_INFO,
                    "medication_info": HealthIntent.MEDICATION_INFO,
                    "wellness_tip": HealthIntent.WELLNESS_TIP,
                    "emergency": HealthIntent.EMERGENCY,
                    "diagnosis_request": HealthIntent.DIAGNOSIS_REQUEST,
                    "prevention_info": HealthIntent.PREVENTION_INFO,
                    "general_health": HealthIntent.GENERAL_HEALTH,
                }

                intent = intent_mapping.get(intent_str, HealthIntent.UNKNOWN)

                # Check for diagnosis request patterns as backup
                if self._is_diagnosis_request(text):
                    intent = HealthIntent.DIAGNOSIS_REQUEST

                return intent, confidence

        except Exception as e:
            print(f"⚠️ Error in intent classification: {e}")

        return HealthIntent.UNKNOWN, 0.5

    def get_intent(self, text: str) -> HealthIntent:
        normalized_text = normalize_hinglish_terms(text.lower())
        for symptom in HINGLISH_SYMPTOMS.keys():
            if symptom in normalized_text:
                return HealthIntent.SYMPTOM_QUERY
        return HealthIntent.GENERAL_HEALTH

    def _extract_medical_entities(self, text: str, language: str) -> List[MedicalEntity]:
        """Extract medical entities using real Sarvam-M API"""

        messages = [
            {
                "role": "system",
                "content": """You are a medical entity extractor. Extract these entity types from healthcare queries:

- symptoms: Detailed descriptions of physical feelings or ailments like fever, headache, cough, chest pain, sore throat, body ache, nausea, dizziness, fatigue, etc. Be specific in identifying the symptom text.
- diseases: diabetes, hypertension, covid, etc.
- medications: paracetamol, metformin, aspirin, etc.
- body_parts: head, chest, stomach, heart, etc.
- medical_terms: blood pressure, sugar level, etc.

Respond ONLY with JSON format:
{"entities": [{"text": "fever", "type": "symptom", "start": 5, "end": 10, "confidence": 0.95}]}"""
            },
            {
                "role": "user",
                "content": f"Extract medical entities from: '{text}'\nLanguage: {language}"
            }
        ]

        entities = []

        try:
            print(f"🔄 Calling Sarvam-M for entity extraction...")
            response = self.sarvam_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )

            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                entity_list = result.get("entities", [])

                for entity_data in entity_list:
                    entity = MedicalEntity(
                        text=entity_data.get("text", ""),
                        entity_type=entity_data.get("type", "unknown"),
                        confidence=entity_data.get("confidence", 0.5),
                        start_pos=entity_data.get("start", 0),
                        end_pos=entity_data.get("end", 0)
                    )
                    entities.append(entity)

        except Exception as e:
            print(f"⚠️ Error in entity extraction: {e}")

        # Augment with keyword matching from symptom knowledge base
        if self.symptom_kb:
            augmented_count = 0
            text_lower = text.lower()
            for symptom_data in self.symptom_kb:
                keywords_to_check = [symptom_data["symptom_name"].lower()] + [kw.lower() for kw in symptom_data.get("keywords", [])]

                for keyword in keywords_to_check:
                    for match in re.finditer(re.escape(keyword), text_lower):
                        start_pos, end_pos = match.span()

                        # Check for overlap with existing LLM-found entities
                        is_covered = False
                        for existing_entity in entities:
                            if existing_entity.entity_type == "symptom":
                                # Simple overlap check: if keyword is within an existing symptom entity's text span
                                if max(start_pos, existing_entity.start_pos) < min(end_pos, existing_entity.end_pos):
                                    is_covered = True
                                    break
                                # Check if the existing entity text contains the keyword (more robust for partial matches by LLM)
                                if keyword in existing_entity.text.lower():
                                    is_covered = True
                                    break

                        if not is_covered:
                            entities.append(MedicalEntity(
                                text=text[start_pos:end_pos], # Use original casing from text
                                entity_type="symptom",
                                confidence=0.75, # Default confidence for keyword match
                                start_pos=start_pos,
                                end_pos=end_pos
                            ))
                            augmented_count += 1
            if augmented_count > 0:
                print(f"ℹ️ Augmented entities with {augmented_count} symptoms from keyword matching.")
        # Apply spelling and phonetic correction to entity texts
        seen = set()
        deduped_entities = []
        for entity in entities:
            original = entity.text
            entity.text = correct_misspelled_entity(entity.text)
            entity.text = phonetic_match(entity.text, tuple(COMMON_MISSPELLINGS.keys()))
            if entity.text != original:
                logger.info(f"🔄 Corrected '{original}' to '{entity.text}'")
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                deduped_entities.append(entity)
        return deduped_entities

    def _is_diagnosis_request(self, text: str) -> bool:
        """Check if text contains diagnosis request patterns"""
        text_lower = text.lower()

        for pattern in self.diagnosis_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def _detect_language(self, text: str) -> str:
        """Detect language of the text"""
        # Simplified language detection
        # In practice, use Sarvam-M's language detection capabilities

        hindi_chars = bool(re.search(r'[\u0900-\u097F]', text))
        tamil_chars = bool(re.search(r'[\u0B80-\u0BFF]', text))
        telugu_chars = bool(re.search(r'[\u0C00-\u0C7F]', text))

        if hindi_chars:
            return "hi-IN"
        elif tamil_chars:
            return "ta-IN"
        elif telugu_chars:
            return "te-IN"
        else:
            return "en-IN"

# Integration with audio capture
def integrate_stt_nlu_pipeline():
    """Example integration of STT + NLU pipeline"""
    try:
        # For now, we'll use a mock transcription since audio_capture module isn't available
        # from src.audio_capture import CleanAudioCapture, SarvamSTTIntegration

        # Initialize NLU processor
        nlu_processor = SarvamMNLUProcessor()

        print("🎤 Starting voice input simulation...")

        # Mock transcribed text for testing
        mock_transcriptions = [
            ("मुझे बुखार है", "hi-IN"),
            ("What are the symptoms of diabetes?", "en-IN"),
            ("Emergency! Chest pain!", "en-IN"),
        ]

        for transcribed_text, source_language in mock_transcriptions:
            print(f"\n📝 Processing: {transcribed_text}")

            # Step 3: NLU with Sarvam-M
            nlu_result = nlu_processor.process_transcription(
                transcribed_text,
                source_language=source_language
            )

            # Step 4: Handle results based on NLU
            if nlu_result.is_emergency:
                print("🚨 EMERGENCY DETECTED!")
                print("Response: Please call emergency services or visit nearest hospital immediately.")
                continue

            if nlu_result.intent == HealthIntent.DIAGNOSIS_REQUEST:
                print("⚠️ DIAGNOSIS REQUEST DETECTED!")
                print("Response: I cannot provide medical diagnosis. Please consult a qualified doctor.")
                continue

            print(f"✅ Intent: {nlu_result.intent.value}")
            print(f"📊 Entities found: {len(nlu_result.entities)}")
            for entity in nlu_result.entities:
                print(f"  - {entity.text} ({entity.entity_type})")

    except Exception as e:
        print(f"❌ Pipeline error: {e}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Check if API key is set
    if not os.getenv("SARVAM_API_KEY"):
        print("❌ Please set SARVAM_API_KEY environment variable")
        print("Get your API key from: https://dashboard.sarvam.ai")
        print("\nExample:")
        print("export SARVAM_API_KEY='your_api_key_here'")
    else:
        integrate_stt_nlu_pipeline()
