from langchain_google_community import SpeechToTextLoader

from agentum import tool


@tool
def transcribe_audio(audio_filepath: str, project_id: str) -> str:
    """
    Transcribes the content of an audio file using Google's Speech-to-Text API.

    This tool is ideal for converting spoken language from an audio file into
    written text for an agent to process.

    Args:
        audio_filepath: The local path to the audio file (e.g., './audio/request.wav').
        project_id: Your Google Cloud Project ID.

    Returns:
        The transcribed text as a string.
    """
    try:
        # The loader handles the interaction with the Google Cloud API
        loader = SpeechToTextLoader(project_id=project_id, file_path=audio_filepath)

        # The result is a list of LangChain 'Document' objects
        documents = loader.load()

        if not documents:
            return "Error: Could not transcribe the audio. The file might be empty or unsupported."

        # We combine the content of all resulting documents into a single text block
        transcription = " ".join(doc.page_content for doc in documents)
        return transcription
    except Exception as e:
        return f"Error during audio transcription: {e}. Ensure you are authenticated with Google Cloud (run 'gcloud auth application-default login') and the Speech-to-Text API is enabled."
