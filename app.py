from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)

# Load pre-trained Stable Diffusion model and CLIP processor for text-to-image generation
stable_diff_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original")
stable_diff_pipe.to("cuda")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.route('/generate_logo', methods=['POST'])
def generate_logo():
    # Extract description from the request
    description = request.json.get('description')

    if not description:
        return jsonify({'error': 'Description is required'}), 400

    # Generate logo using diffusion model
    generated_image = generate_image_from_description(description)
    
    # Save the generated image to a file or directly return in response
    generated_image.save("generated_logo.png")

    return jsonify({'message': 'Logo generated successfully!', 'image_url': 'generated_logo.png'}), 200

def generate_image_from_description(description):
    # Use the Stable Diffusion model to generate the logo image
    image = stable_diff_pipe(description).images[0]
    return image

if __name__ == '__main__':
    app.run(debug=True)


