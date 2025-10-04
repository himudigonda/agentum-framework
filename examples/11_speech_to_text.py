"""
This example showcases Agentum's new Speech-to-Text (STT) capabilities.

The workflow uses the `transcribe_audio` tool to convert a spoken request
from an audio file into text. An agent then uses this text to perform a
task (in this case, a web search).

To run this example:
1. Make sure you are authenticated with Google Cloud:
   `gcloud auth application-default login`
2. Enable the Speech-to-Text API in your Google Cloud project.
3. Set your GOOGLE_CLOUD_PROJECT_ID in your .env file.
4. Download the sample audio file:
   `curl -o request.wav "https://storage.googleapis.com/agentum-examples/request.wav"`
5. Run the script.
"""

import os

from dotenv import load_dotenv

from agentum import (
    Agent,
    GoogleLLM,
    State,
    Workflow,
    search_web_tavily,
    transcribe_audio,
)
from agentum.core.config import settings

load_dotenv()


class VoiceRequestState(State):
    audio_filepath: str
    transcription: str = ""
    research_result: str = ""


researcher = Agent(
    name="VoiceResearcher",
    system_prompt="You are a helpful assistant. You first understand a user's request by transcribing it, then you search the web to find an answer.",
    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"),
    tools=[search_web_tavily, transcribe_audio],
)
voice_workflow = Workflow(name="Voice_Request_Workflow", state=VoiceRequestState)
voice_workflow.add_task(
    name="transcribe_request",
    tool=transcribe_audio,
    inputs={
        "audio_filepath": "{audio_filepath}",
        "project_id": settings.GOOGLE_CLOUD_PROJECT_ID,
    },
    output_mapping={"transcription": "output"},
)
voice_workflow.add_task(
    name="conduct_research",
    agent=researcher,
    instructions="Perform a web search for the following request: {transcription}",
    output_mapping={"research_result": "output"},
)
voice_workflow.set_entry_point("transcribe_request")
voice_workflow.add_edge("transcribe_request", "conduct_research")
voice_workflow.add_edge("conduct_research", voice_workflow.END)
if __name__ == "__main__":
    audio_file = "request.wav"
    if not os.path.exists(audio_file):
        print("Audio file not found. Please download it by running:")
        print(
            'curl -o request.wav "https://storage.googleapis.com/agentum-examples/request.wav"'
        )
    else:
        initial_state = {"audio_filepath": audio_file}
        print(f"🚀 Starting workflow for audio file: {audio_file}")
        final_state = voice_workflow.run(initial_state)
        print("\n--- Transcription ---")
        print(f"Text from audio: '{final_state['transcription']}'")
        print("\n--- Research Result ---")
        print(final_state["research_result"])
        print("\n✅ Speech-to-Text workflow completed successfully!")
