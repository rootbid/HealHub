import streamlit as st
import streamlit.components.v1 as components
import os
import json
import time # For polling audio capture status
import numpy as np # For checking audio data (though not directly used in this version)
from dotenv import load_dotenv
from typing import Optional
import re
from streamlit_mic_recorder import mic_recorder
import soundfile as sf
import io
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Adjust import paths
try:
    from src.nlu_processor import SarvamMNLUProcessor, HealthIntent, NLUResult
    from src.response_generator import HealHubResponseGenerator
    from src.symptom_checker import SymptomChecker
    from src.audio_capture import AudioCleaner # Import audio modules
    from src.utils import HealHubUtilities
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.nlu_processor import SarvamMNLUProcessor, HealthIntent, NLUResult
    from src.response_generator import HealHubResponseGenerator
    from src.symptom_checker import SymptomChecker
    from src.audio_capture import AudioCleaner
    from src.utils import HealHubUtilities

# --- Environment and API Key Setup ---
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# --- Session State Initialization ---
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'current_language_display' not in st.session_state: 
    st.session_state.current_language_display = 'English'
if 'current_language_code' not in st.session_state: 
    st.session_state.current_language_code = 'en-IN'
if 'text_query_input_area' not in st.session_state:
    st.session_state.text_query_input_area = ""

# Symptom Checker states
if 'symptom_checker_active' not in st.session_state:
    st.session_state.symptom_checker_active = False
if 'symptom_checker_instance' not in st.session_state:
    st.session_state.symptom_checker_instance = None
if 'pending_symptom_question_data' not in st.session_state:
    st.session_state.pending_symptom_question_data = None

# Voice Input states
if 'voice_input_stage' not in st.session_state:
    # Stages: None, "arming", "recording", "transcribing", "processing_stt"
    st.session_state.voice_input_stage = None 
if 'audio_capturer' not in st.session_state: 
    st.session_state.audio_capturer = None
if 'captured_audio_data' not in st.session_state:
    st.session_state.captured_audio_data = None
if 'cleaned_audio_data' not in st.session_state:
    st.session_state.cleaned_audio_data = None
if "captured_audio_sample_rate" not in st.session_state:
    st.session_state.captured_audio_sample_rate = 48000

# --- Language Mapping ---
LANGUAGE_MAP = {
    "English": "en-IN", 
    "हिन्दी (Hindi)": "hi-IN", 
    "বাংলা (Bengali)": "bn-IN", 
    "मराठी (Marathi)": "mr-IN", 
    "ಕನ್ನಡ (Kannada)": "kn-IN",
    "தமிழ் (Tamil)": "ta-IN",
    "తెలుగు (Telugu)": "te-IN",
    "മലയാളം (Malayalam)": "ml-IN",
}

DISPLAY_LANGUAGES = list(LANGUAGE_MAP.keys())



# --- Helper Functions ---
def add_message_to_conversation(role: str, content: str, lang_code: Optional[str] = None):
    message = {"role": role, "content": content}
    if lang_code and role == "user":
        message["lang"] = lang_code 
    st.session_state.conversation.append(message)


# --- Streamlit UI ---
def main_ui():
    st.set_page_config(page_title="HealHub", layout="wide", initial_sidebar_state="collapsed")

    if not firebase_admin._apps:
        # Load credentials from Streamlit secrets
        try:
            cred_dict = {
                "type": st.secrets["firebase"]["type"],
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'), # Important for newlines
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": st.secrets["firebase"]["auth_uri"],
                "token_uri": st.secrets["firebase"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
                "universe_domain": st.secrets["firebase"]["universe_domain"]
            }
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Error initializing Firebase: {e}")
            # st.info("Please ensure your .streamlit/secrets.toml is correctly configured with Firebase credentials.")
            st.stop() # Stop the app if Firebase fails to initialize

    db = firestore.client()

    def store_feedback(feedback_text, user_email, ml_generated_text, full_conversation):
        try:
            # Prepare feedback data
            feedback_data = {
                "timestamp": datetime.now(), # Store current timestamp
                "user_email": user_email if user_email.strip() else "Anonymous",
                "feedback_text": feedback_text,
                "ml_generated_text": ml_generated_text,
                "full_conversation": full_conversation,
            }

            # Add data to Firestore
            # Create a new document in the 'feedback' collection
            db.collection("feedback").add(feedback_data)
            st.success("Thank you for your feedback! It has been submitted.")
            return True
            # Optionally clear the form
            feedback_text = ""
            user_email = ""
        except Exception as e:
            st.error(f"An error occurred while submitting feedback: {e}")
            return False



    # st.caption("Your AI healthcare companion. Supporting English and Popular Indic Languages.")
    header_css = """
        <style>
            /* Hide the main header */
            header {visibility: hidden;}

            /* Remove the reserved space for the header */
            .block-container {
                padding-top: 1rem;
            }

            /* Optional: hide hamburger menu and footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            .feedback-button {
                background-color: transparent;
                border: none;
                color: var(--text-color);
                font-size: 1.2rem;
                cursor: pointer;
                margin-right: 0rem;
            }
            .feedback-button:hover {
                opacity: 0.8;
            }
            .feedback-container {
                display: flex;
                justify-content: flex-end;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
            }
        </style>
        """
    st.markdown(header_css, unsafe_allow_html=True)
    
    if not SARVAM_API_KEY: 
        st.error("🚨 SARVAM_API_KEY not found. Please set it in your .env file for the application to function.")
        st.stop()

    col1, col2 = st.columns([3, 9])
    with col1:
        st.title("💬 HealHub")
        st.caption("Your AI healthcare companion. Supporting English and Popular Indic Languages.")
        st.markdown(f"<div style='height: 40px;'></div>", unsafe_allow_html=True)
        selected_lang_display = st.selectbox(
            "Select Language / भाषा चुनें / ভাষা নির্বাচন করুন:",
            options=DISPLAY_LANGUAGES,
            index=DISPLAY_LANGUAGES.index(st.session_state.current_language_display),
            key='language_selector_widget' 
        )
        if selected_lang_display != st.session_state.current_language_display:
            st.session_state.current_language_display = selected_lang_display
            st.session_state.current_language_code = LANGUAGE_MAP[selected_lang_display]
            st.session_state.conversation = [] 
            st.session_state.symptom_checker_active = False; st.session_state.symptom_checker_instance = None; st.session_state.pending_symptom_question_data = None
            st.session_state.voice_input_stage = None
            st.rerun()

        current_lang_code_for_query = st.session_state.current_language_code
        spinner_placeholder = st.empty()

        # All functions which needs time to process and will utilize spinner placeholder for loading screen
        def process_and_display_response(user_query_text: str, lang_code: str):
            if not SARVAM_API_KEY:
                st.error("API Key not configured.")
                add_message_to_conversation("system", "Error: API Key not configured.")
                st.session_state.voice_input_stage = None # Reset voice stage on error
                return

            nlu_processor = SarvamMNLUProcessor(api_key=SARVAM_API_KEY)
            response_gen = HealHubResponseGenerator(api_key=SARVAM_API_KEY)
            util = HealHubUtilities(api_key=SARVAM_API_KEY)
            user_lang = st.session_state.current_language_code
            try:
                # User message is now added *before* calling this function for both text and voice.
                # So, this function should not add the user message again.
                
                with spinner_placeholder.info("🧠 Thinking..."):
                    nlu_output: NLUResult = nlu_processor.process_transcription(user_query_text, source_language=lang_code)
                    if nlu_output.intent == HealthIntent.SYMPTOM_QUERY and not nlu_output.is_emergency:
                        st.session_state.symptom_checker_active = True
                        st.session_state.symptom_checker_instance = SymptomChecker(nlu_result=nlu_output, api_key=SARVAM_API_KEY)
                        st.session_state.symptom_checker_instance.prepare_follow_up_questions()
                        st.session_state.pending_symptom_question_data = st.session_state.symptom_checker_instance.get_next_question()
                        if st.session_state.pending_symptom_question_data:
                            question_to_ask_raw = st.session_state.pending_symptom_question_data['question']
                            symptom_context_raw = st.session_state.pending_symptom_question_data['symptom_name']
                            question_to_ask_translated = util.translate_text(question_to_ask_raw, user_lang)
                            symptom_context_translated = util.translate_text(symptom_context_raw, user_lang)
                            add_message_to_conversation("assistant", f"{question_to_ask_translated}: {symptom_context_translated}")
                        else:
                            generate_and_display_assessment()
                    else:
                        bot_response = response_gen.generate_response(user_query_text, nlu_output)
                        translated_bot_response = util.translate_text(bot_response, user_lang)
                        add_message_to_conversation("assistant", translated_bot_response)
                        st.session_state.symptom_checker_active = False
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Standardized error message
                add_message_to_conversation("system", f"Sorry, an error occurred while processing your request. Please try rephrasing or try again later. (Details: {str(e)})")
                st.session_state.symptom_checker_active = False # Reset states on error
                st.session_state.symptom_checker_instance = None
                st.session_state.pending_symptom_question_data = None
            finally:
                st.session_state.voice_input_stage = None # Always reset voice stage after processing or error

        def handle_follow_up_answer(answer_text: str):
            util = HealHubUtilities(api_key=SARVAM_API_KEY)
            user_lang = st.session_state.current_language_code
            if st.session_state.symptom_checker_instance and st.session_state.pending_symptom_question_data:
                # Add user's follow-up answer to conversation log
                add_message_to_conversation("user", answer_text, lang_code=st.session_state.current_language_code.split('-')[0])
                
                question_asked = st.session_state.pending_symptom_question_data['question']
                symptom_name = st.session_state.pending_symptom_question_data['symptom_name']
                with spinner_placeholder.info("Recording answer..."):
                    st.session_state.symptom_checker_instance.record_answer(symptom_name, question_asked, answer_text)
                    st.session_state.pending_symptom_question_data = st.session_state.symptom_checker_instance.get_next_question()
                if st.session_state.pending_symptom_question_data:
                    question_to_ask_raw = st.session_state.pending_symptom_question_data['question']
                    symptom_context_raw = st.session_state.pending_symptom_question_data['symptom_name']
                    question_to_ask_translated = util.translate_text(question_to_ask_raw, user_lang)
                    symptom_context_translated = util.translate_text(symptom_context_raw, user_lang)
                    add_message_to_conversation("assistant", f"{symptom_context_translated}: {question_to_ask_translated}")
                else:
                    generate_and_display_assessment()
            else: 
                st.warning("No pending question to answer or symptom checker not active.")
                st.session_state.symptom_checker_active = False
            st.session_state.voice_input_stage = None # Reset voice stage

        # New callback function for text submission
        def handle_text_submission():
            user_input = str(st.session_state.text_query_input_area).strip() # Read from session state key
            current_lang_code = st.session_state.current_language_code

            if not user_input: # Do nothing if input is empty
                return

            # Add the current user input to conversation log REGARDLESS of whether it's new or follow-up
            
            if st.session_state.symptom_checker_active and st.session_state.pending_symptom_question_data:
                # handle_follow_up_answer will process the answer.
                # It should NOT add the user message again as it's already added above.
                handle_follow_up_answer(user_input) 
            else: 
                add_message_to_conversation("user", user_input, lang_code=current_lang_code.split('-')[0])
                if st.session_state.symptom_checker_active: # Reset if symptom checker was active but no pending q
                    st.session_state.symptom_checker_active = False 
                    st.session_state.symptom_checker_instance = None
                    st.session_state.pending_symptom_question_data = None
                # process_and_display_response will process the new query.
                # It should NOT add the user message again.
                process_and_display_response(user_input, current_lang_code)
            
            st.session_state.text_query_input_area = "" # Clear the text area state for next render
            # If called from a non-button context that needs immediate UI update, rerun might be needed.

        def generate_and_display_assessment():
            util = HealHubUtilities(api_key=SARVAM_API_KEY)
            user_lang = st.session_state.current_language_code
            if st.session_state.symptom_checker_instance:
                with spinner_placeholder.info("🔬 Generating preliminary assessment..."):
                    assessment = st.session_state.symptom_checker_instance.generate_preliminary_assessment()
                    try:
                        assessment_str = f"<h4> {util.translate_text('Preliminary Health Assessment', user_lang)}:</h4>\n\n"
                        assessment_str += f"**{util.translate_text('Summary', user_lang)}:** {util.translate_text(assessment.get('assessment_summary', 'N/A'), user_lang)}\n\n"
                        assessment_str += f"**{util.translate_text('Suggested Severity', user_lang)}:** {util.translate_text(assessment.get('suggested_severity', 'N/A'), user_lang)}\n\n"
                        assessment_str += f"**{util.translate_text('Recommended Next Steps', user_lang)}:**\n"
                        next_steps = assessment.get('recommended_next_steps', 'N/A')
                        if isinstance(next_steps, list): 
                            for step in next_steps: assessment_str += f"- {util.translate_text(step, user_lang)}\n"
                        elif isinstance(next_steps, str): # This is the block to modify
                            ### Replace the original problematic f-string line here
                            # Split on punctuation marks (., !, ?) followed by whitespace
                            sentences = re.split(r'(?<=[.!?])\s+', next_steps.strip())
                            # Add bullet to each sentence
                            temp_steps = '\n- '.join(sentences).strip()
                            # remove leading bullet if present (e.g. if next_steps started with punctuation)
                            temp_steps = temp_steps.lstrip('- ')
                            # Append to assessment_str
                            assessment_str += f"{util.translate_text(temp_steps, user_lang)}\n"
                        else: 
                            assessment_str += f"- {util.translate_text('N/A', user_lang)}\n"
                        warnings = assessment.get('potential_warnings')
                        if warnings and isinstance(warnings, list) and len(warnings) > 0 :
                            assessment_str += f"\n**{util.translate_text('Potential Warnings', user_lang)}:**\n"
                            for warning in warnings: assessment_str += f"- {util.translate_text(warning, user_lang)}\n"
                        kb_points = assessment.get('relevant_kb_triage_points')
                        if kb_points and isinstance(kb_points, list) and len(kb_points) > 0:
                            assessment_str += f"\n**{util.translate_text('Relevant Triage Points from Knowledge Base', user_lang)}:**\n"
                            for point in kb_points: assessment_str += f"- {util.translate_text(point, user_lang)}\n"
                        assessment_str += f"\n\n**{util.translate_text('Disclaimer', user_lang)}:** {util.translate_text(assessment.get('disclaimer', 'Always consult a doctor for medical advice.'), user_lang)}"
                        add_message_to_conversation("assistant", assessment_str)
                    except Exception as e:
                        st.error(f"Error formatting assessment: {e}")
                        try:
                            raw_assessment_json = json.dumps(assessment, indent=2)
                            add_message_to_conversation("assistant", f"Could not format assessment. Raw data:\n```json\n{raw_assessment_json}\n```")
                        except Exception as json_e:
                            add_message_to_conversation("assistant", f"Could not format or serialize assessment: {json_e}")
                st.session_state.symptom_checker_active = False
                st.session_state.symptom_checker_instance = None
                st.session_state.pending_symptom_question_data = None
            st.session_state.voice_input_stage = None # Reset voice stage

        # Capture and Process audio
        if st.session_state.captured_audio_data is not None:
            with spinner_placeholder.info("Cleaning the captured audio..."):
                with io.BytesIO(st.session_state.captured_audio_data) as buffer:
                    data, sr = sf.read(buffer)
                # Clean audio
                cleaner = AudioCleaner()
                cleaned_data, cleaned_sr = cleaner.get_cleaned_audio(data, sr)
            ### To test captured and cleaned audio
            # audio_buffer = io.BytesIO()
            # sf.write(audio_buffer, cleaned_data, cleaned_sr, format='WAV')
            # audio_buffer.seek(0)
            # st.audio(audio_buffer.getvalue(), format="audio/wav")
            st.session_state.cleaned_audio_data = cleaned_data
            st.session_state.captured_audio_sample_rate = cleaned_sr
            st.session_state.voice_input_stage = "processing_stt"
        
        if st.session_state.voice_input_stage == "processing_stt":
            if st.session_state.cleaned_audio_data is not None:
                util = HealHubUtilities(api_key=SARVAM_API_KEY)
                lang_for_stt = st.session_state.current_language_code 
                try:
                    with spinner_placeholder.info("Transcribing audio..."):
                        stt_result = util.transcribe_audio(
                            st.session_state.cleaned_audio_data, sample_rate=st.session_state.captured_audio_sample_rate, source_language=lang_for_stt
                        )
                    transcribed_text = stt_result.get("transcription")
                    if lang_for_stt != stt_result.get("language_detected"):
                        if lang_for_stt == "en-IN":
                            transcribed_text = util.translate_text_to_english(transcribed_text)
                        else:
                            transcribed_text = util.translate_text(transcribed_text, lang_for_stt)
                    if transcribed_text and transcribed_text.strip():
                        add_message_to_conversation("user", transcribed_text, lang_code=lang_for_stt.split('-')[0])
                        process_and_display_response(transcribed_text, lang_for_stt) 
                    else:
                        add_message_to_conversation("system", "⚠️ STT failed to transcribe audio or returned empty. Please try again.")
                except Exception as e:
                    st.error(f"STT Error: {e}")
                    add_message_to_conversation("system", f"Sorry, an error occurred during voice transcription. Please try again. (Details: {e})")
                st.session_state.captured_audio_data = None 
                st.session_state.cleaned_audio_data = None 
                st.session_state.voice_input_stage = None 
                st.rerun()
            else: 
                st.session_state.voice_input_stage = None
                st.rerun()

    with col2:

        def handle_good_feedback(idx, content):
            store_feedback("It's a good feedback", "", content, st.session_state.conversation)
    
        st.markdown("### Conversation")
        chat_container = st.container(height=350) 
        with chat_container:
            util = HealHubUtilities(api_key=SARVAM_API_KEY)
            user_lang = st.session_state.current_language_code
            for idx, msg_data in enumerate(st.session_state.conversation):
                role = msg_data.get("role", "system") 
                content = msg_data.get("content", "")
                lang_display = msg_data.get('lang', st.session_state.current_language_code.split('-')[0])

                if role == "user":
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.3rem;">                            
                            <div style="
                                background-color: rgba(183,183,183,0.25);
                                padding: 0.4rem 0.7rem;
                                border-radius: 0.6rem;
                                max-width: 75%;
                                text-align: right;
                                word-wrap: break-word;">{content}</div>
                            <div style="
                                width: 32px; height: 32px;
                                border-radius: 50%;
                                border: 2px solid rgba(183,183,183, 0.5);
                                background-color: transparent;
                                display: flex; align-items: center; justify-content: center;
                                font-size: 18px;">🧑‍💻</div>
                        </div>
                    """, unsafe_allow_html=True)
                elif role == "assistant":
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.3rem;">
                            <div style="
                                width: 32px; height: 32px;
                                border-radius: 50%;
                                border: 2px solid rgba(183,183,183, 0.5);
                                background-color: transparent;
                                display: flex; align-items: center; justify-content: center;
                                font-size: 18px;">⚕️</div>
                            <div style="
                                background-color: rgba(83,83,83,0.25);
                                padding: 0.4rem 0.7rem;
                                border-radius: 0.6rem;
                                max-width: 75%;
                                text-align: left;
                                word-wrap: break-word;">{content}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    clutter, col1, col2, col3, clutter = st.columns([1.75, 1, 1, 1, 30])
                    audio_bytes = None
                    good_feedback = False
                    with col1:
                        if st.button("👍", key=f"good_{idx}", type="tertiary", help="Good response"):
                            good_feedback = True
                    with col2:
                        if st.button("👎", key=f"bad_{idx}", type="tertiary", help="Bad response"):
                            st.session_state[f"negetive_feedback_{idx}"] = True

                    with col3:
                        if st.button("🔊", key=f"read_{idx}", type="tertiary", help="Read aloud"):
                            with spinner_placeholder.info("Synthesizing speech..."):
                                audio_bytes = util.synthesize_speech(content, user_lang)
                            
                    if good_feedback is True:
                        handle_good_feedback(idx, content)
                    if audio_bytes is not None:
                        st.audio(audio_bytes, format="audio/wav")
                    if st.session_state.get(f"negetive_feedback_{idx}", False):
                        with st.expander("Tell us why you disliked this response:", expanded=True):
                            user_email = st.text_input("Your Email Id", key=f"user_email_{idx}")
                            feedback_text = st.text_area("Your feedback", key=f"feedback_text_{idx}")
                            if st.button("Submit Feedback", key=f"submit_feedback_{idx}"):
                                feedback_response = store_feedback(feedback_text, user_email, content, st.session_state.conversation)
                                if feedback_response is True:
                                    st.session_state[f"negetive_feedback_{idx}"] = False  # Reset if needed after submission
                                    st.rerun()
                else:
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.3rem;">
                            <div style="
                                width: 32px; height: 32px;
                                border-radius: 50%;
                                border: 2px solid rgba(255, 255, 255, 0.3);
                                background-color: transparent;
                                display: flex; align-items: center; justify-content: center;
                                font-size: 18px;">ℹ️</div>
                            <div style="
                                background-color: transparent;
                                padding: 0.4rem 0.7rem;
                                border-radius: 0.6rem;
                                max-width: 75%;
                                text-align: left;
                                word-wrap: break-word;">{content}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
            st.markdown("""
                <style>
                    button[kind="tertiary"] {
                        background: none !important;
                        border: none !important;
                        color: inherit !important;
                        padding: 0 !important;
                        margin: 0 0 0 0 !important;
                        font-size: 0rem !important;
                        line-height: 0 !important;
                        width: auto !important;
                        height: auto !important;
                    }
                </style>
            """, unsafe_allow_html=True)
        
        

        is_recording = st.session_state.voice_input_stage == "recording"
        
        # Define column width ratios for the input area, send button, and voice recording button
        
        input_label = "Type your answer here...(Ctrl.+Enter to send)" if st.session_state.symptom_checker_active and st.session_state.pending_symptom_question_data else "Type your health query here...(Ctrl.+Enter to send)"
    
        # Text area widget - its current value is stored in st.session_state.text_query_input_area due to its key
        st.text_area(input_label, height=70, key="text_query_input_area", disabled=is_recording, on_change=handle_text_submission)
        # st.text_input(input_label, key="text_query_input_area", disabled=is_recording, on_change=handle_text_submission)
        
        # user_input = st.text_input("Type your query or use voice:", key="user_input")
        
        COLUMN_WIDTHS = [1, 1]
        col21, col22 = st.columns(COLUMN_WIDTHS)

        with col21:
            st.button(
                "📤 Send", 
                use_container_width=True, 
                key="send_button_widget", # Key can be kept if useful for other logic, or removed
                disabled=is_recording,
                on_click=handle_text_submission # Assign the callback
            )

        with col22:
            audio = mic_recorder(
                start_prompt="🎙️ Record",
                stop_prompt="⏹️ Stop",
                just_once=True,  # Only returns audio once after recording
                use_container_width=True,
                format="wav",    # Or "webm" if you prefer
                key="voice_recorder"
            )
        
        if audio:
            st.session_state.captured_audio_data = audio['bytes']
            st.rerun()
    # The old `if send_button and user_query_text_from_area:` block is now removed,
    # as its logic is handled by the handle_text_submission callback.


if __name__ == "__main__":
    main_ui()
