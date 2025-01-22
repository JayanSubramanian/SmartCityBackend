import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel
import numpy as np

# Load environment variables
load_dotenv()

# API keys for Generative AI (Google Gemini)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-1.5-pro')

# Flask setup
app = Flask(__name__)
CORS(app)

# SCARA robot kinematics
l1 = 100  # Length of the first arm (in px or chosen unit)
l2 = 100  # Length of the second arm (in px or chosen unit)

# Fixed bin location on the canvas (for example, at (x_bin, y_bin))
x_bin, y_bin = 150, 150  # Change these values to the desired bin location

def inverse_kinematics(x, y):
    """Compute the joint angles for the SCARA robot."""
    D = (x**2 + y**2 - l1**2 - l2**2)
    
    if D < 0:
        raise ValueError("The point is unreachable.")

    theta2 = np.arccos(D / (2 * l1 * l2))  # Calculate the second joint angle
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))  # First joint angle
    
    return theta1, theta2

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read and process the image
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or 'JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Generate waste classification using Gemini
    response = model.generate_content([
        {
            "inline_data": {
                "data": img_str,  # Base64 image data
                "mime_type": file.content_type or "image/jpeg"  # MIME type of the image
            }
        },
        "Classify the waste shown in this image into one of the following categories: "
        "1. Plastic (e.g., bottles, bags, containers, wrappers), "
        "2. Paper (e.g., newspapers, cardboard, office paper, magazines), "
        "3. Organic (e.g., food waste, garden waste, leaves, fruit peels), "
        "4. Metal (e.g., cans, aluminum foil, metal scraps, wires), "
        "5. Glass (e.g., bottles, jars, broken glass), "
        "6. Electronic (e.g., old phones, chargers, computers, circuit boards), "
        "7. Textiles (e.g., clothes, fabric scraps, rugs), "
        "8. Hazardous (e.g., batteries, chemicals, medical waste, paint cans), "
        "9. Rubber (e.g., tires, rubber bands, hoses), "
        "10. Construction debris (e.g., bricks, cement, tiles, wood planks), "
        "11. Mixed waste (e.g., unsegregated trash, multiple categories in one item), "
        "or any other specific category not listed here. "
        "Provide a concise explanation for your classification, highlighting why the item fits into the category based on its visible properties, material, or context."
    ])

    # Parse the response
    classification_result = response.text
    print("Gemini Response:", classification_result)

    return jsonify({
        'classification': classification_result,
    })

if __name__ == '__main__':
    app.run(debug=True)
