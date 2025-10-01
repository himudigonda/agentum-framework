"""
This is a capstone example demonstrating a complete, end-to-end voice assistant.

The workflow performs the following steps:
1.  **Listens**: Transcribes a user's spoken question from an audio file using the `transcribe_audio` tool.
2.  **Thinks**: An agent generates a text-based answer to the transcribed question.
3.  **Speaks**: Converts the agent's text answer back into a spoken audio file using the `text_to_speech` tool.

This showcases how Agentum seamlessly orchestrates STT, agentic logic, and TTS
to create a fully voice-interactive system.

To run this example:
1.  Authenticate with Google Cloud: `gcloud auth application-default login`
2.  Enable both the Speech-to-Text and Text-to-Speech APIs in your project.
3.  Set GOOGLE_CLOUD_PROJECT_ID in your .env file.
4.  Ensure the sample audio file 'request.wav' is present.
5.  Run the script. An 'response.mp3' file will be generated.
"""

import os

from dotenv import load_dotenv

# MODIFICATION: Import settings
from agentum import Agent, GoogleLLM, State, Workflow, text_to_speech, transcribe_audio
from agentum.config import settings

load_dotenv()


class VoiceAssistantState(State):
    input_audio_path: str
    output_audio_path: str
    question_text: str = ""
    answer_text: str = ""


assistant_agent = Agent(
    name="HelpfulAssistant",
    system_prompt="You are a helpful and friendly assistant. You provide concise, clear answers to questions.",
    # MODIFICATION: Use settings object
    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"),
)

voice_assistant_workflow = Workflow(
    name="Voice_Assistant_Workflow", state=VoiceAssistantState
)

voice_assistant_workflow.add_task(
    name="transcribe_question",
    tool=transcribe_audio,
    inputs={
        "audio_filepath": "{input_audio_path}",
        # MODIFICATION: We can now omit 'project_id' as the tool will get it from settings.
        # This simplifies the workflow definition.
        # "project_id": settings.GOOGLE_CLOUD_PROJECT_ID,
    },
    output_mapping={"question_text": "output"},
)

voice_assistant_workflow.add_task(
    name="generate_answer",
    agent=assistant_agent,
    instructions="Answer the following question: {question_text}",
    output_mapping={"answer_text": "output"},
)

voice_assistant_workflow.add_task(
    name="speak_answer",
    tool=text_to_speech,
    inputs={
        "text_to_speak": "{answer_text}",
        "output_filepath": "{output_audio_path}",
    },
)

voice_assistant_workflow.set_entry_point("transcribe_question")
voice_assistant_workflow.add_edge("transcribe_question", "generate_answer")
voice_assistant_workflow.add_edge("generate_answer", "speak_answer")
voice_assistant_workflow.add_edge("speak_answer", voice_assistant_workflow.END)

if __name__ == "__main__":
    input_file = "request.wav"
    output_file = "response.mp3"

    if not os.path.exists(input_file):
        print(
            f"Input audio file not found. Please run the command from example 11 to download it."
        )
    else:
        initial_state = {
            "input_audio_path": input_file,
            "output_audio_path": output_file,
        }

        print("🚀 Starting end-to-end Voice Assistant workflow...")
        final_state = voice_assistant_workflow.run(initial_state)

        print("\n--- Interaction Summary ---")
        print(f"User (from audio): '{final_state['question_text']}'")
        print(f"Assistant (to audio): '{final_state['answer_text']}'")

        if os.path.exists(output_file):
            print(f"\n✅ Success! Assistant's spoken response saved to: {output_file}")
        else:
            print(f"\n❌ Error: Output audio file was not created.")
