"""
This example demonstrates Agentum's enhanced vision capabilities, showing
how an agent can analyze a LOCAL image file from your computer.

The workflow:
1.  Takes a local file path to an image.
2.  The Agentum engine automatically reads, encodes, and includes the image
    in the prompt to the multi-modal LLM.
3.  The agent analyzes the image content to answer a question.

To run this example:
1.  Download a sample image file by running the following command in your terminal:
    `curl -o cityscape.jpg "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?q=80&w=2070&auto=format&fit=crop"`
2.  Run the script.
"""
import os
from dotenv import load_dotenv
from agentum import Agent, GoogleLLM, State, Workflow
load_dotenv()

class LocalVisionState(State):
    question: str
    image_path: str
    description: str = ''
visual_analyst = Agent(name='LocalVisualAnalyst', system_prompt='You are an expert at analyzing images from local files. Describe what you see in detail.', llm=GoogleLLM(api_key=os.getenv('GOOGLE_API_KEY'), model='gemini-2.5-flash-lite'))
local_vision_workflow = Workflow(name='Local_Image_Analysis_Workflow', state=LocalVisionState)
local_vision_workflow.add_task(name='analyze_local_image', agent=visual_analyst, instructions='{question}', output_mapping={'description': 'output'})
local_vision_workflow.set_entry_point('analyze_local_image')
local_vision_workflow.add_edge('analyze_local_image', local_vision_workflow.END)
if __name__ == '__main__':
    local_image = 'cityscape.jpg'
    if not os.path.exists(local_image):
        print(f'Image file not found. Please download it by running:')
        print('curl -o cityscape.jpg "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?q=80&w=2070&auto=format&fit=crop"')
    else:
        initial_state = {'question': 'Describe this cityscape. What time of day is it and what is the general atmosphere?', 'image_path': local_image}
        print(f"🚀 Starting local image analysis for file: {initial_state['image_path']}")
        final_state = local_vision_workflow.run(initial_state)
        print('\n--- Local Image Analysis Result ---')
        print(final_state['description'])
        print('\n✅ Local vision workflow completed successfully!')