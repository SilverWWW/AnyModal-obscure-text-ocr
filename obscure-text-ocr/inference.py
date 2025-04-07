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
import copy
import random

# Load language model and tokenizer for trained and base model

llm_tokenizer_trained_1, llm_model_trained_1 = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='filler',
    quantized = False,
    use_peft = False
)

llm_tokenizer_trained_2, llm_model_trained_2 = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='filler',
    quantized = False,
    use_peft = False
)
llm_tokenizer_trained_3, llm_model_trained_3 = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='filler',
    quantized = False,
    use_peft = False
)

llm_tokenizer_base, llm_model_base = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='filler',
    quantized = False,
    use_peft = False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm_hidden_size = llm.get_hidden_size(llm_tokenizer_trained_1, llm_model_trained_1)

llm_model_trained_1.to(device)
llm_model_trained_2.to(device)
llm_model_trained_3.to(device)
llm_model_base.to(device)


# Dataset configuration
dataset_name = "MiXaiLL76/TextOCR_OCR"

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/siglip-so400m-patch14-384', use_peft=False)

# Load dataset
ds = vision.ImageDataset(dataset_name, image_processor, name = None, split = 'test')

# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

# Initialize Fine Tuned Model 1
multimodal_model_trained_1 = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer_trained_1,
    language_model=llm_model_trained_1,
    prompt_text="The text in the image is: ")

multimodal_model_trained_1._load_model('obscure_text_ocr_again_1')
multimodal_model_trained_1.eval()

# Initialize Fine Tuned Model 2
multimodal_model_trained_2 = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer_trained_2,
    language_model=llm_model_trained_2,
    prompt_text="The text in the image is: ")

multimodal_model_trained_2._load_model('obscure_text_ocr_again_2')
multimodal_model_trained_2.eval()

# Initialize Fine Tuned Model 3
multimodal_model_trained_3 = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer_trained_3,
    language_model=llm_model_trained_3,
    prompt_text="The text in the image is: ")

multimodal_model_trained_3._load_model('obscure_text_ocr_again_3')
multimodal_model_trained_3.eval()

# Initialize Base Model
multimodal_model_base = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer_base,
    language_model=llm_model_base,
    prompt_text="The text in the image is: ")

multimodal_model_base.eval()

# Generate captions comparatively with the trained and base model

os.makedirs("temp", exist_ok=True)

test_samples = 1000
correct_generations_trained_1 = 0
correct_generations_trained_2 = 0
correct_generations_trained_3 = 0
correct_generations_base = 0

# Generate test_samples # of random sample indices
sampled_indices = random.sample(range(1, len(ds)), test_samples)

with open(f"temp/generations.txt", "w") as f:
    f.write(f"OCR Results Comparison\n")
    f.write(f"====================\n\n")

for i, sample_idx in enumerate(sampled_indices):

    print(f"Sample {i+1}")

    sample = ds[sample_idx]
    
    caption = sample['text']

    input_for_trained_1 = copy.deepcopy(sample['input'])
    input_for_trained_2 = copy.deepcopy(sample['input'])
    input_for_trained_3 = copy.deepcopy(sample['input'])
    input_for_base = copy.deepcopy(sample['input'])

    generated_caption_1 = multimodal_model_trained_1.generate(input_for_trained_1, max_new_tokens=15, do_sample = True, num_beams = 3)
    generated_caption_2 = multimodal_model_trained_2.generate(input_for_trained_2, max_new_tokens=15, do_sample = True, num_beams = 3)
    generated_caption_3 = multimodal_model_trained_3.generate(input_for_trained_3, max_new_tokens=15, do_sample = True, num_beams = 3)

    generated_caption_base = multimodal_model_base.generate(input_for_base, max_new_tokens=15, do_sample = True, num_beams = 3)

    if generated_caption_1 == caption:
        correct_generations_trained_1 += 1

    if generated_caption_2 == caption:
        correct_generations_trained_2 += 1

    if generated_caption_3 == caption:
        correct_generations_trained_3 += 1

    if generated_caption_base == caption:
        correct_generations_base += 1

    with open(f"temp/generations.txt", "a") as f:
        f.write(f"Actual Caption for Sample {sample_idx}: {caption}\n")

        f.write(f"Trained Model 1 - Generated Caption: {generated_caption_1}\n")
        f.write(f"Trained Model 2 - Generated Caption: {generated_caption_2}\n")
        f.write(f"Trained Model 3 - Generated Caption: {generated_caption_3}\n")

        f.write(f"Base Model      - Generated Caption: {generated_caption_base}\n\n")

with open(f"temp/generations.txt", "a") as f:
        f.write(f"--------------------------------------------------\n")
        f.write(f"Trained model 1 accuracy: {correct_generations_trained_1 / test_samples}\n")
        f.write(f"Trained model 2 accuracy: {correct_generations_trained_2 / test_samples}\n")
        f.write(f"Trained model 3 accuracy: {correct_generations_trained_3 / test_samples}\n")

        f.write(f"Base model accuracy: {correct_generations_base / test_samples}")