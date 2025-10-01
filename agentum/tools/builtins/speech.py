from google.cloud import texttospeech
from langchain_google_community import SpeechToTextLoader

from agentum import tool
from agentum.config import settings


@tool
def transcribe_audio(audio_filepath: str, project_id: str | None = None) -> str:
    """
    Transcribes the content of an audio file using Google's Speech-to-Text API.

    This tool is ideal for converting spoken language from an audio file into
    written text for an agent to process.

    Args:
        audio_filepath: The local path to the audio file (e.g., './audio/request.wav').
        project_id: Your Google Cloud Project ID. If not provided, will use the global setting.

    Returns:
        The transcribed text as a string.
    """
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
    """
    Converts a string of text into a spoken audio file using Google's
    Text-to-Speech API.

    Args:
        text_to_speak: The text content to be converted to speech.
        output_filepath: The local path to save the resulting audio file (e.g., './audio/response.mp3').

    Returns:
        A confirmation string with the path to the saved audio file.
    """
    try:
        client = texttospeech.TextToSpeechClient()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

        # Build the voice request, select a language code ("en-US") and the ssml
        # voice gender ("neutral")
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        # Select the type of audio file you want
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # The response's audio_content is binary.
        with open(output_filepath, "wb") as out:
            # Write the response to the output file.
            out.write(response.audio_content)

        return f"Successfully saved synthesized audio to {output_filepath}"
    except Exception as e:
        return f"Error during text-to-speech synthesis: {e}. Ensure you are authenticated with Google Cloud (run 'gcloud auth application-default login') and the Text-to-Speech API is enabled."
