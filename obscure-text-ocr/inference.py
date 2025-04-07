# pip install huggingface_hub transformers torch schedulefree peft datasets bitsandbytes sentencepiece
# pip install -U pillow
import llm
import anymodal
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
import matplotlib
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download, snapshot_download
import matplotlib.pyplot as plt

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='filler',
    quantized = False,
    use_peft = False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

llm_model.to(device)

# Dataset configuration
dataset_name = "MiXaiLL76/TextOCR_OCR"

# Load vision model components
# image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/siglip-so400m-patch14-384', use_peft=False)

# Load dataset
ds = vision.ImageDataset(dataset_name, image_processor, name = None, split = 'test')

# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)


# Initialize Fine Tuned Model
multimodal_model_trained = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    prompt_text="The text in the image is: ")

multimodal_model_trained._load_model('obscure_text_ocr_again_3')
multimodal_model_trained.eval()

# Initialize Base Model
multimodal_model_base = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    prompt_text="The text in the image is: ")

multimodal_model_base._load_model('obscure_text_ocr_dummy_1')
multimodal_model_base.eval()

# Generate captions comparatively with the trained and base model

os.makedirs("temp", exist_ok=True)

test_samples = 1000
correct_generations_trained = 0
correct_generations_base = 0

with open(f"temp/generations.txt", "w") as f:
    f.write(f"OCR Results Comparison\n")
    f.write(f"====================\n\n")

for i in range(test_samples):
    sample_idx = i
    sample = ds[sample_idx]
    
    caption = sample['text']
    generated_caption_1 = multimodal_model_trained.generate(sample['input'], max_new_tokens=120, do_sample = True, num_beams = 3)
    generated_caption_2 = multimodal_model_base.generate(sample['input'], max_new_tokens=120, do_sample = True, num_beams = 3)

    if generated_caption_1 == caption:
        correct_generations_trained += 1

    if generated_caption_2 == caption:
        correct_generations_base += 1

    with open(f"temp/generations.txt", "a") as f:
        f.write(f"Actual Caption: {caption}\n")
        f.write(f"Trained Model - Generated Caption: {generated_caption_1}\n")
        f.write(f"Base Model    - Generated Caption: {generated_caption_2}\n\n")

with open(f"temp/generations.txt", "a") as f:
        f.write(f"--------------------------------------------------\n")
        f.write(f"Trained model accuracy: {correct_generations_trained / test_samples}")
        f.write(f"Base model accuracy: {correct_generations_base / test_samples}")