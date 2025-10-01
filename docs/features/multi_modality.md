# Multi-Modal Capabilities: Vision

Agentum allows your agents to "see" by analyzing images from both web URLs and local files. The engine automatically detects when an image is part of the state and constructs the correct multi-modal prompt for the LLM.

## Analyzing a Web Image (URL)

To analyze an image from the web, simply include an `image_url` field in your workflow's `State`.

```python
from agentum import Agent, GoogleLLM, State, Workflow

# The 'image_url' field is the key
class VisionState(State):
    question: str
    image_url: str
    description: str = ""

visual_analyst = Agent(
    name="VisualAnalyst",
    system_prompt="You are an expert at analyzing images.",
    llm=GoogleLLM(model="gemini-2.5-flash-lite")
)

workflow = Workflow(name="Web_Image_Analysis", state=VisionState)
workflow.add_task(
    name="analyze",
    agent=visual_analyst,
    instructions="{question}", # The image is passed implicitly
    output_mapping={"description": "output"},
)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", workflow.END)

# Run the workflow
result = workflow.run({
    "question": "Describe this city skyline.",
    "image_url": "https://images.unsplash.com/photo-1542051841857-5f90071e7989"
})
print(result["description"])
```

## Analyzing a Local Image File

To analyze an image from your computer, use the `image_path` field in your `State`. The Agentum engine will handle opening, encoding, and embedding the image for you.

```python
# The 'image_path' field is the key
class LocalVisionState(State):
    question: str
    image_path: str # Path to a local file, e.g., "./images/photo.jpg"
    description: str = ""

# The rest of the workflow definition is identical.
workflow = Workflow(name="Local_Image_Analysis", state=LocalVisionState)
workflow.add_task(
    name="analyze",
    agent=visual_analyst,
    instructions="{question}",
    output_mapping={"description": "output"},
)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", workflow.END)

# Run with a local image
result = workflow.run({
    "question": "What is happening in this picture?",
    "image_path": "cityscape.jpg"
})
print(result["description"])
```

## Priority System

When both `image_url` and `image_path` are present in the state, Agentum prioritizes local files over URLs:

1. **First Priority:** `image_path` (local file)
2. **Second Priority:** `image_url` (web URL)

This ensures that local files take precedence when both are available.

## Supported Image Formats

Agentum supports all common image formats through Python's `mimetypes` module:

- **JPEG/JPG**
- **PNG**
- **GIF**
- **WebP**
- **BMP**
- **TIFF**

## Error Handling

The engine gracefully handles common issues:

- **Missing files:** Shows a warning and continues without the image
- **Invalid URLs:** Shows a warning and continues without the image
- **Unsupported formats:** Shows a warning and continues without the image
- **Permission errors:** Shows a warning and continues without the image

## Real-World Use Cases

Vision capabilities enable powerful applications:

- **Content moderation:** Analyze uploaded images for inappropriate content
- **Document analysis:** Extract text and data from scanned documents
- **Product cataloging:** Automatically categorize and describe products
- **Medical imaging:** Assist in analyzing X-rays, MRIs, and other medical images
- **Security monitoring:** Analyze surveillance footage for anomalies
- **Social media:** Generate captions and descriptions for user-uploaded images

## Best Practices

1. **Use descriptive questions:** Be specific about what you want the agent to analyze
2. **Provide context:** Include relevant background information in your prompts
3. **Handle errors gracefully:** Always check if the image was successfully processed
4. **Optimize image size:** Large images may take longer to process
5. **Test with different formats:** Ensure your workflow works with various image types

## Examples

- **Web Image Analysis:** See `examples/09_vision_analysis.py`
- **Local Image Analysis:** See `examples/14_local_image_analysis.py`
- **Multi-Modal Research:** Combine vision with web search for comprehensive analysis

## Technical Details

The vision system works by:

1. **Detection:** Checking for `image_url` or `image_path` in the state
2. **Loading:** Reading the image file or fetching from URL
3. **Encoding:** Converting to base64 format with proper MIME type
4. **Construction:** Building multi-modal message content for the LLM
5. **Processing:** Sending the combined text and image to the model

This seamless integration means you don't need to worry about the technical details - just include the image in your state and let Agentum handle the rest!
