from google.cloud import texttospeech
from langchain_google_community import SpeechToTextLoader

from agentum import tool
from agentum.config import settings


@tool
def transcribe_audio(audio_filepath: str, project_id: str | None = None) -> str:
    try:
        proj_id = project_id or settings.GOOGLE_CLOUD_PROJECT_ID
        if not proj_id:
            return "Error: A Google Cloud Project ID was not provided and is not set in the environment (GOOGLE_CLOUD_PROJECT_ID)."

        loader = SpeechToTextLoader(project_id=proj_id, file_path=audio_filepath)

        documents = loader.load()

        if not documents:
            return "Error: Could not transcribe the audio. The file might be empty or unsupported."

        transcription = " ".join(doc.page_content for doc in documents)
        return transcription
    except Exception as e:
        return f"Error during audio transcription: {e}. Ensure you are authenticated with Google Cloud (run 'gcloud auth application-default login') and the Speech-to-Text API is enabled."


@tool
def text_to_speech(text_to_speak: str, output_filepath: str) -> str:
    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_filepath, "wb") as out:
            out.write(response.audio_content)

        return f"Successfully saved synthesized audio to {output_filepath}"
    except Exception as e:
        return f"Error during text-to-speech synthesis: {e}. Ensure you are authenticated with Google Cloud (run 'gcloud auth application-default login') and the Text-to-Speech API is enabled."
