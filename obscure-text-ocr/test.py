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

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/siglip-so400m-patch14-384', use_peft=False)

dataset_name = "MiXaiLL76/TextOCR_OCR"
full_train_dataset = vision.ImageDataset(dataset_name, image_processor, name=None, split='train')

# Calculate which indices were NOT used in your training
subset_ratio = 0.5
total_length = len(full_train_dataset)
used_indices = set(range(int(subset_ratio * total_length)))

# Get the indices that weren't used
all_indices = set(range(total_length))
unused_indices = list(all_indices - used_indices)

# Create a new subset with only the unused indices
unseen_train_dataset = torch.utils.data.Subset(full_train_dataset, unused_indices)
batch_size = 4
test_loader = DataLoader(unseen_train_dataset, batch_size=batch_size, shuffle=True)

test_size = len(test_loader)
print(f"Test size: {test_size}")

# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)


# Initialize MultiModalModel
multimodal_model = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    prompt_text="The text in the image is: ")

multimodal_model._load_model('obscure_text_ocr_3')

multimodal_model.eval()
test_losses = []

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(test_loader), desc=f"Test", leave=False):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = multimodal_model(batch)
        test_losses.append(loss.item())

        if batch_idx > 1000:
            break
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Test Loss: {avg_test_loss:.4f}")