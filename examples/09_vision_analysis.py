"""
Demonstrates Agentum's multi-modal vision capabilities.

The workflow takes a URL to an image and a question, and uses an agent
to analyze the image content to answer the question. This showcases how
Agentum can now handle both text and image data in a single, seamless workflow.
"""

import os

from dotenv import load_dotenv

from agentum import Agent, GoogleLLM, State, Workflow
from agentum.config import settings

load_dotenv()


class VisionState(State):
    question: str
    image_url: str
    description: str = ""


visual_analyst = Agent(
    name="VisualAnalyst",
    system_prompt="You are an expert at analyzing images and describing what you see in detail. You answer questions based on the provided image.",
    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"),
)
vision_workflow = Workflow(name="Image_Analysis_Workflow", state=VisionState)
vision_workflow.add_task(
    name="analyze_image",
    agent=visual_analyst,
    instructions="{question}",
    output_mapping={"description": "output"},
)
vision_workflow.set_entry_point("analyze_image")
vision_workflow.add_edge("analyze_image", vision_workflow.END)
if __name__ == "__main__":
    initial_state = {
        "question": "Describe this scene. What city is it, and what is the overall mood?",
        "image_url": "https://images.unsplash.com/photo-1542051841857-5f90071e7989?q=80&w=2070&auto=format&fit=crop",
    }
    print(f"🚀 Starting vision analysis for image: {initial_state['image_url']}")
    final_state = vision_workflow.run(initial_state)
    print("\n--- Image Analysis Result ---")
    print(final_state["description"])
    if (
        "tokyo" in final_state["description"].lower()
        and "skyline" in final_state["description"].lower()
    ):
        print(
            "\n✅ Vision workflow completed successfully! The agent correctly identified the scene."
        )
    else:
        print(
            "\n⚠️  Workflow completed, but the description may not be accurate. Check the output."
        )
