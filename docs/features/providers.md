# LLM Providers

Agentum is designed to be provider-agnostic, allowing you to easily switch between different Large Language Models without changing your core agent or workflow logic. We currently support:

- **Google** (`GoogleLLM`): For Gemini models.
- **Anthropic** (`AnthropicLLM`): For Claude models.
- **OpenAI** (`OpenAILLM`): For GPT models.

## Switching Between Providers

Switching models is as simple as importing and instantiating a different provider class. The `Agent` class accepts any of these provider objects.

The example below shows how you can toggle between all three major providers with a one-line code change.

```python
import os
from dotenv import load_dotenv
from agentum import Agent, State, Workflow, GoogleLLM, AnthropicLLM, OpenAILLM

load_dotenv()

# --- This is the only part you need to change ---

# Option 1: Google Gemini (Default)
llm_provider = GoogleLLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-lite",
)

# Option 2: Anthropic Claude (uncomment to use)
# llm_provider = AnthropicLLM(
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
#     model="claude-3-5-sonnet-20240620",
# )

# Option 3: OpenAI GPT-4o-mini (uncomment to use)
# llm_provider = OpenAILLM(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model="gpt-4o-mini",
# )

# ------------------------------------------------

# The rest of your code remains exactly the same.
class StoryState(State):
    topic: str
    story: str = ""

storyteller = Agent(
    name="Storyteller",
    system_prompt="You are a master storyteller.",
    llm=llm_provider, # The selected provider is passed here
)

workflow = Workflow(name="Story_Workflow", state=StoryState)
workflow.add_task(
    name="write_story",
    agent=storyteller,
    instructions="Write a story about: {topic}",
    output_mapping={"story": "output"},
)
workflow.set_entry_point("write_story")
workflow.add_edge("write_story", workflow.END)

# Run with any provider
result = workflow.run({"topic": "a robot learning to paint"})
print(result["story"])
```

## Provider-Specific Features

### Google Gemini (`GoogleLLM`)

**Strengths:**
- Excellent multi-modal capabilities (vision, audio)
- Fast response times
- Cost-effective for most use cases
- Strong reasoning abilities

**Best for:**
- Multi-modal applications
- General-purpose tasks
- Cost-sensitive projects
- Real-time applications

**Configuration:**
```python
google_llm = GoogleLLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-lite",  # Fast and efficient
    temperature=0.7,
    max_tokens=1000
)
```

### Anthropic Claude (`AnthropicLLM`)

**Strengths:**
- Exceptional reasoning and analysis
- Long context windows
- Strong safety features
- Excellent for complex tasks

**Best for:**
- Complex analysis and reasoning
- Long-form content generation
- Research and academic work
- Safety-critical applications

**Configuration:**
```python
anthropic_llm = AnthropicLLM(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20240620",  # Latest and most capable
    temperature=0.7,
    max_tokens=1000
)
```

### OpenAI GPT (`OpenAILLM`)

**Strengths:**
- Mature ecosystem
- Extensive tool integration
- Reliable performance
- Strong code generation

**Best for:**
- Code generation and debugging
- Creative writing
- General-purpose applications
- Integration with existing OpenAI tools

**Configuration:**
```python
openai_llm = OpenAILLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",  # Cost-effective option
    temperature=0.7,
    max_tokens=1000
)
```

## Provider Comparison

| Feature | Google Gemini | Anthropic Claude | OpenAI GPT |
|---------|---------------|------------------|------------|
| **Multi-modal** | ✅ Excellent | ❌ Text only | ✅ Good |
| **Reasoning** | ✅ Good | ✅ Excellent | ✅ Good |
| **Speed** | ✅ Fast | ⚠️ Moderate | ✅ Fast |
| **Cost** | ✅ Low | ⚠️ Moderate | ⚠️ Moderate |
| **Context Length** | ✅ Long | ✅ Very Long | ✅ Long |
| **Code Generation** | ✅ Good | ✅ Good | ✅ Excellent |

## Best Practices

### 1. Choose the Right Provider for Your Use Case

- **Multi-modal tasks:** Use Google Gemini
- **Complex reasoning:** Use Anthropic Claude
- **Code generation:** Use OpenAI GPT
- **General purpose:** Any provider works well

### 2. Handle API Keys Securely

```python
# Always use environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# Never hardcode API keys
llm = GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"))
```

### 3. Implement Fallback Strategies

```python
def create_llm_with_fallback():
    """Create an LLM with fallback providers."""
    try:
        return GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception:
        try:
            return AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except Exception:
            return OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"))
```

### 4. Monitor Usage and Costs

- Set up usage alerts for each provider
- Monitor token consumption
- Use appropriate models for your use case
- Consider rate limits and quotas

## Advanced Configuration

### Custom Parameters

Each provider supports additional parameters:

```python
# Google Gemini with custom settings
google_llm = GoogleLLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-lite",
    temperature=0.3,  # More deterministic
    max_tokens=2000,  # Longer responses
    top_p=0.9,  # Nucleus sampling
    top_k=40  # Top-k sampling
)

# Anthropic Claude with custom settings
anthropic_llm = AnthropicLLM(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20240620",
    temperature=0.1,  # Very deterministic
    max_tokens=4000,  # Very long responses
    stop_sequences=["\n\n---"]  # Custom stop sequences
)

# OpenAI GPT with custom settings
openai_llm = OpenAILLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.5,  # Balanced creativity
    max_tokens=1500,  # Medium responses
    frequency_penalty=0.1,  # Reduce repetition
    presence_penalty=0.1  # Encourage new topics
)
```

## Examples

- **Multi-Provider Example:** See `examples/08_multiple_llms.py`
- **Provider-Specific Features:** Check individual provider documentation
- **Fallback Strategies:** Implement error handling for production use

## Troubleshooting

### Common Issues

1. **API Key Errors:** Ensure your API keys are correctly set in the `.env` file
2. **Rate Limiting:** Implement exponential backoff for production use
3. **Model Availability:** Check if the model is available in your region
4. **Token Limits:** Monitor token usage and adjust `max_tokens` accordingly

### Getting Help

- Check the provider-specific documentation
- Review the examples in the `examples/` directory
- Test with different providers to find the best fit for your use case
