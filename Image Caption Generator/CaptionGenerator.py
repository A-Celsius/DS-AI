from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import gradio as gr

model_name = "Salesforce/blip-image-captioning-base"

caption_processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_captions(image, num_captions=5,size=(512, 512)):
    image = image.resize(size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = caption_processor(image, return_tensors='pt').to(device)
    
    caption_ids = model.generate(
        **pixel_values, 
        max_length=30,
        num_beams=5,
        num_return_sequences=num_captions,
        temperature=1.0
    )
    
    captions = [
        caption_processor.decode(ids, skip_special_tokens=True) 
        for ids in caption_ids
    ]
    
    return captions

from gradio.components import Image, Textbox,Slider

interface = gr.Interface(
    fn=generate_captions,
    inputs=[
        Image(type="pil", label="Input Image"),
        Slider(minimum=1, maximum=5, step=1, label="Number of Captions")
    ],
    outputs=Textbox(type="text", label="Captions"),
    title="Assignment 1",
    description="AI tool that creates captions based on the image provided by the user.",
)

interface.launch()