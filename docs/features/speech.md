# Speech Capabilities: STT and TTS

Agentum provides built-in tools to create end-to-end voice-enabled workflows. Your agents can now listen to user requests and speak their responses.

These tools use Google's high-quality Speech APIs and require you to be authenticated (`gcloud auth application-default login`) and have the APIs enabled in your project.

## Prerequisites

Before using speech capabilities, ensure you have:

1. **Google Cloud Authentication:**
   ```bash
   gcloud auth application-default login
   ```

2. **API Keys Enabled:**
   - Speech-to-Text API
   - Text-to-Speech API

3. **Project ID:** Set `GOOGLE_CLOUD_PROJECT_ID` in your `.env` file

4. **Dependencies Installed:**
   ```bash
   pip install langchain-google-community google-cloud-texttospeech
   ```

## Speech-to-Text (`transcribe_audio`)

The `transcribe_audio` tool converts an audio file into text.

| Argument | Type | Description |
|---|---|---|
| `audio_filepath` | `str` | The local path to the audio file (e.g., `request.wav`). |
| `project_id` | `str` | Your Google Cloud Project ID. |

### Example Usage

```python
from agentum import transcribe_audio, Workflow, State

class TranscribeState(State):
    audio_in: str
    text_out: str = ""

workflow = Workflow(name="Transcribe_Test", state=TranscribeState)
workflow.add_task(
    name="listen",
    tool=transcribe_audio,
    inputs={
        "audio_filepath": "{audio_in}",
        "project_id": "your-gcp-project-id"
    },
    output_mapping={"text_out": "output"}
)
workflow.set_entry_point("listen")
workflow.add_edge("listen", workflow.END)

# Run the workflow
result = workflow.run({
    "audio_in": "user_request.wav"
})
print(f"Transcribed text: {result['text_out']}")
```

### Supported Audio Formats

- **WAV** (recommended)
- **MP3**
- **FLAC**
- **OGG**
- **M4A**
- **AAC**

### Best Practices for STT

1. **Use clear audio:** Minimize background noise
2. **Optimal length:** Keep audio files under 60 seconds for best results
3. **Sample rate:** Use 16kHz or higher for better accuracy
4. **Format:** WAV files typically provide the best results

## Text-to-Speech (`text_to_speech`)

The `text_to_speech` tool converts a string of text into an MP3 audio file.

| Argument | Type | Description |
|---|---|---|
| `text_to_speak` | `str` | The text you want to convert to speech. |
| `output_filepath`| `str` | The path to save the generated MP3 file (e.g., `response.mp3`). |

### Example Usage

```python
from agentum import text_to_speech, Workflow, State

class SpeakState(State):
    text_in: str

workflow = Workflow(name="Speak_Test", state=SpeakState)
workflow.add_task(
    name="speak",
    tool=text_to_speech,
    inputs={
        "text_to_speak": "{text_in}",
        "output_filepath": "response.mp3"
    }
)
workflow.set_entry_point("speak")
workflow.add_edge("speak", workflow.END)

# Run the workflow
workflow.run({
    "text_in": "Hello! This is your AI assistant speaking."
})
```

### TTS Configuration

The tool uses Google's Text-to-Speech API with these default settings:

- **Language:** English (en-US)
- **Voice:** Neutral gender
- **Audio Format:** MP3
- **Sample Rate:** 22050 Hz

## End-to-End Voice Assistant

By combining these tools with an agent, you can create a complete voice assistant that listens, thinks, and speaks.

### Complete Voice Assistant Example

```python
import os
from dotenv import load_dotenv
from agentum import (
    Agent, GoogleLLM, State, Workflow,
    text_to_speech, transcribe_audio
)

load_dotenv()

# 1. Define the state for voice interaction
class VoiceAssistantState(State):
    input_audio_path: str
    output_audio_path: str
    question_text: str = ""
    answer_text: str = ""

# 2. Create the assistant agent
assistant_agent = Agent(
    name="VoiceAssistant",
    system_prompt="You are a helpful voice assistant. Provide concise, clear answers.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"),
)

# 3. Build the complete voice workflow
voice_workflow = Workflow(name="Voice_Assistant", state=VoiceAssistantState)

# Task 1: Listen - Transcribe the input audio
voice_workflow.add_task(
    name="listen",
    tool=transcribe_audio,
    inputs={
        "audio_filepath": "{input_audio_path}",
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
    },
    output_mapping={"question_text": "output"},
)

# Task 2: Think - Generate a text answer
voice_workflow.add_task(
    name="think",
    agent=assistant_agent,
    instructions="Answer this question: {question_text}",
    output_mapping={"answer_text": "output"},
)

# Task 3: Speak - Convert the answer to speech
voice_workflow.add_task(
    name="speak",
    tool=text_to_speech,
    inputs={
        "text_to_speak": "{answer_text}",
        "output_filepath": "{output_audio_path}",
    },
)

# 4. Define the control flow
voice_workflow.set_entry_point("listen")
voice_workflow.add_edge("listen", "think")
voice_workflow.add_edge("think", "speak")
voice_workflow.add_edge("speak", voice_workflow.END)

# 5. Run the complete voice assistant
if __name__ == "__main__":
    result = voice_workflow.run({
        "input_audio_path": "user_question.wav",
        "output_audio_path": "assistant_response.mp3"
    })
    
    print(f"User asked: '{result['question_text']}'")
    print(f"Assistant responded: '{result['answer_text']}'")
    print(f"Audio response saved to: {result['output_audio_path']}")
```

## Real-World Use Cases

Speech capabilities enable powerful voice applications:

### 1. **Voice-Enabled Customer Support**
- Transcribe customer calls
- Generate automated responses
- Provide voice-based support

### 2. **Accessibility Applications**
- Voice-to-text for users with mobility issues
- Text-to-speech for users with visual impairments
- Voice navigation for applications

### 3. **Educational Tools**
- Language learning with pronunciation feedback
- Interactive voice-based tutorials
- Audio content generation

### 4. **Smart Home Integration**
- Voice commands for home automation
- Audio notifications and alerts
- Voice-based device control

### 5. **Content Creation**
- Podcast generation from text
- Audio book creation
- Voice-over for videos

## Error Handling

The speech tools include comprehensive error handling:

```python
# Example of error handling in your workflow
try:
    result = transcribe_audio("audio.wav", "project-id")
    if "Error" in result:
        print(f"Transcription failed: {result}")
    else:
        print(f"Transcription successful: {result}")
except Exception as e:
    print(f"Tool execution failed: {e}")
```

### Common Error Scenarios

1. **Authentication Issues:**
   - Solution: Run `gcloud auth application-default login`

2. **API Not Enabled:**
   - Solution: Enable Speech-to-Text and Text-to-Speech APIs in Google Cloud Console

3. **Invalid Audio Format:**
   - Solution: Convert to supported format (WAV recommended)

4. **File Not Found:**
   - Solution: Check file path and permissions

5. **Project ID Missing:**
   - Solution: Set `GOOGLE_CLOUD_PROJECT_ID` in `.env` file

## Performance Optimization

### 1. **Audio File Optimization**
- Use WAV format for best quality
- Keep files under 60 seconds
- Use 16kHz sample rate minimum

### 2. **Text Optimization for TTS**
- Keep responses concise
- Use clear, simple language
- Avoid special characters that might cause issues

### 3. **Workflow Optimization**
- Process audio files in parallel when possible
- Cache frequently used audio files
- Use appropriate file compression

## Examples

- **Speech-to-Text:** See `examples/11_speech_to_text.py`
- **Voice Assistant:** See `examples/12_voice_assistant.py`
- **Multi-Modal Integration:** Combine speech with vision and web search

## Troubleshooting

### Common Issues

1. **"Your default credentials were not found"**
   - Run: `gcloud auth application-default login`

2. **"API not enabled"**
   - Enable Speech-to-Text and Text-to-Speech APIs in Google Cloud Console

3. **"Project ID not found"**
   - Set `GOOGLE_CLOUD_PROJECT_ID` in your `.env` file

4. **Audio file not found**
   - Check file path and ensure file exists
   - Verify file permissions

### Getting Help

- Check Google Cloud documentation for Speech APIs
- Review the examples in the `examples/` directory
- Test with simple audio files first
- Verify your Google Cloud setup
