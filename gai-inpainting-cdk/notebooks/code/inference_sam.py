import numpy as np
import torch
import os
import json
import io
import cv2
import time
from transformers import SamModel, SamProcessor
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the processor with the pre-trained model
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")

def model_fn(model_dir):
    """
    Load the pre-trained model from the specified directory.

    Args:
        model_dir (str): Directory containing the pre-trained model files.

    Returns:
        model: Loaded pre-trained model.
    """
    print("Executing model_fn from inference.py ...")
    env = os.environ
    model = SamModel.from_pretrained("facebook/sam-vit-large")
    model.to(device)
    return model

def input_fn(request_body, request_content_type):
    """
    Preprocess the input data.

    Args:
        request_body: Input data from the request.
        request_content_type (str): Content type of the request.

    Returns:
        inputs: Preprocessed input data.
    """
    print("Executing input_fn from inference.py ...")
    inputs = []
    if request_content_type:
        # Load image array from request body
        img_array = np.load(io.BytesIO(request_body), allow_pickle=True)
        img = Image.fromarray(img_array)

        # Define input points for the processor
        input_points = [[[np.array(img.size)/2]]]

        # Preprocess the image using the processor
        inputs = processor(img, input_points=input_points, return_tensors="pt")
        inputs = inputs.to(device)
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return inputs
    
def predict_fn(input_data, model):
    """
    Perform the prediction using the input data and the loaded model.

    Args:
        input_data: Preprocessed input data.
        model: Loaded pre-trained model.

    Returns:
        result: Prediction output.
    """
    print("Executing predict_fn from inference.py ...")
    result = []
    with torch.no_grad():
        # Perform the prediction using the model
        result = model(**input_data)

        # Post-process the predicted masks
        result = processor.image_processor.post_process_masks(result.pred_masks.cpu(), input_data["original_sizes"].cpu(), input_data["reshaped_input_sizes"].cpu())

        if torch.cuda.is_available():
            # Empty the GPU cache and collect garbage
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    return result
        
def output_fn(prediction_output, content_type):
    """
    Process the prediction output and prepare the response.

    Args:
        prediction_output: Prediction output.
        content_type (str): Desired content type for the response.

    Returns:
        str: Response in the specified content type.
    """
    print("Executing output_fn from inference.py ...")
    masks = np.transpose(prediction_output[0][0, :, :, :].numpy(), [1, 2, 0]).astype(np.uint8) * 255
    mask_list = masks.tolist()
    return json.dumps(mask_list)
