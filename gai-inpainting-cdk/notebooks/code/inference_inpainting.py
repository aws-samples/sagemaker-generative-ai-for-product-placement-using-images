from PIL import Image
from PIL import ImageFilter
import torch
import os
import json
import io
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
import transformers
import albumentations as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformation
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.4, scale_limit=(-0.4, -0.4), rotate_limit=0, p=1, border_mode=1, interpolation=4,
                       shift_limit_y=[0.1, 0.25])
])

torch_dtype = torch.float32
if torch.cuda.is_available():
    torch_dtype = torch.float16

model_id_inpainting = "runwayml/stable-diffusion-inpainting"
model_id_t2i = "runwayml/stable-diffusion-v1-5"

def find_contour(image):
    """
    Finds contours in the given image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the contours and the thresholded image.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    # Return the contours
    return contours, thresh

def get_centroid(contour):
    """
    Calculates the centroid of a contour.

    Args:
        contour (numpy.ndarray): The input contour.

    Returns:
        tuple: A tuple containing the centroid coordinates (x, y).
    """
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx = 0
        cy = 0
    return cx, cy

def model_fn(model_dir):
    """
    Loads the model and returns the model object.

    Args:
        model_dir (str): The directory where the model is stored.

    Returns:
        list: A list containing the loaded models.
    """
    print("Executing model_fn from inference.py ...")
    model = []
    env = os.environ

    # Load the StableDiffusionInpaintPipeline and StableDiffusionPipeline models
    pipeline_inpainting = StableDiffusionInpaintPipeline.from_pretrained(model_id_inpainting, torch_dtype=torch_dtype)
    pipeline_t2i = StableDiffusionPipeline.from_pretrained(model_id_t2i, torch_dtype=torch_dtype)

    pipeline_inpainting = pipeline_inpainting.to(device)
    pipeline_t2i = pipeline_t2i.to(device)

    model = [pipeline_inpainting, pipeline_t2i]
    return model

def input_fn(request_body, request_content_type):
    """
    Parses the input request body and returns the input data for prediction.

    Args:
        request_body (str): The request body containing the input data.
        request_content_type (str): The content type of the request.

    Returns:
        dict: A dictionary containing the parsed input data.
    """
    print("Executing input_fn from inference.py ...")
    request_body = json.loads(request_body)
    inputs = {}
    if request_content_type:
        # Load the image and mask from the request body
        img_array = np.uint8(np.array(request_body["image"]))
        img = Image.fromarray(img_array).convert('RGB').resize((512, 512))
        inputs["image"] = img
        msk_array = np.uint8(np.array(request_body["mask"]))
        msk = Image.fromarray(np.array(Image.fromarray(msk_array).convert('L')) == 0).convert('RGB').resize((512, 512))
        inputs["mask"] = msk
        inputs["prompt_fr"] = request_body["prompt_fr"]
        inputs["prompt_bg"] = request_body["prompt_bg"]
        inputs["negative_prompt"] = request_body["negative_prompt"]
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return inputs

def predict_fn(input_data, model):
    """
    Runs the prediction on the input data using the loaded model.

    Args:
        input_data (dict): The input data for prediction.
        model (list): The loaded models.

    Returns:
        dict: The prediction results.
    """
    print("Executing predict_fn from inference.py ...")
    result = {}
    pipeline_inpainting, pipeline_t2i = model[0], model[1]
    with torch.no_grad():
        image_bg = pipeline_t2i(input_data["prompt_bg"], guidance_scale=10).images[0]

        # Apply image transformation and blend with the background
        transformed_image_mask = transform(image=np.array(input_data["image"]), mask=np.array(input_data["mask"]))
        image_np = np.array(image_bg)
        idx = np.where(transformed_image_mask["mask"] == 0)
        image_np[idx[0], idx[1], :] = transformed_image_mask["image"][idx[0], idx[1], :]
        transformed_image_mask["image"] = image_np

        # Find the contours and centroid of the mask
        contours, ret = find_contour(transformed_image_mask["mask"])
        cx, cy = get_centroid(contours[0])
        max_min_cy = (contours[0][:, 0, 1].max() - contours[0][:, 0, 1].min()) // 3

        # Create masks for the lower and upper portions of the object
        anchor_point = int(np.random.uniform(0, max_min_cy))
        anti_mask = np.copy(transformed_image_mask["mask"])
        anti_mask[(cy - anchor_point):, :] = 0
        anti_mask_lower = 255 - (transformed_image_mask["mask"] - anti_mask)

        anti_mask = np.copy(transformed_image_mask["mask"])
        anti_mask[:(cy - anchor_point), :] = 0
        anti_mask_upper = 255 - (transformed_image_mask["mask"] - anti_mask)
        anti_mask_upper = Image.fromarray(255 - anti_mask_upper).filter(ImageFilter.GaussianBlur(4))
        new_mask_ori = Image.fromarray(transformed_image_mask["mask"])
        new_mask = new_mask_ori.filter(ImageFilter.GaussianBlur(8))

        transformed_image_mask["mask"] = 255 - anti_mask_lower
        transformed_image_mask["mask"] = np.array(
            Image.fromarray(transformed_image_mask["mask"]).filter(ImageFilter.GaussianBlur(4)))

        image_fr = Image.fromarray(transformed_image_mask["image"])

        # Run the inpainting pipeline on the foreground image
        image_fr = pipeline_inpainting(
            prompt=input_data["prompt_fr"],
            image=image_fr,
            mask_image=Image.fromarray(transformed_image_mask["mask"]),
            num_inference_steps=100,
            guidance_scale=8.5,
            negative_prompt=input_data["negative_prompt"]
        ).images[0]

        # Run the inpainting pipeline on the background image
        image_fr = pipeline_inpainting(
            prompt=input_data["prompt_bg"],
            image=image_fr,
            mask_image=anti_mask_upper,
            num_inference_steps=100,
            guidance_scale=8.5,
            negative_prompt=input_data["negative_prompt"]
        ).images[0]

        result["image"] = image_fr
        result["background"] = image_bg
        result["mask"] = Image.fromarray(((1 - np.array(new_mask) / 255) * (1) * 255).astype(np.uint8))

        scale = (1 - np.array(new_mask_ori)/255)*(1 - np.array(new_mask_ori)/255)
        image_np = scale*transformed_image_mask["image"] + (1 - scale)*image_fr
        result["postprocess_image"] = Image.fromarray(image_np.astype(np.uint8))


        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return result

def output_fn(prediction_output, content_type):
    """
    Converts the prediction output to the specified content type and returns it.

    Args:
        prediction_output (dict): The prediction output.
        content_type (str): The desired content type.

    Returns:
        str: The prediction output converted to the specified content type.
    """
    print("Executing output_fn from inference.py ...")
    all_list = [np.array(prediction_output["image"]).tolist(), np.array(prediction_output["background"]).tolist(),
                np.array(prediction_output["mask"]).tolist(), np.array(prediction_output["postprocess_image"]).tolist()]

    return json.dumps(all_list)
