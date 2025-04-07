import llm
import anymodal
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
from torch.amp import GradScaler

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='filler',   
    use_peft=True
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Dataset configuration
dataset_name = "MiXaiLL76/TextOCR_OCR"

# Load vision model components
# image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/siglip-so400m-patch14-384', use_peft=False)

# Load dataset
full_train_dataset = vision.ImageDataset(dataset_name, image_processor, name = None, split = 'train')
full_test_dataset = vision.ImageDataset(dataset_name, image_processor, name = None, split = 'test')

train_val_ratio = 0.8  # 80% train, 20% validation from original train split
train_size = int(train_val_ratio * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

all_indices = list(range(len(full_train_dataset)))
torch.manual_seed(42)
random_indices = torch.randperm(len(all_indices)).tolist()

train_indices = random_indices[:train_size]
val_indices = random_indices[train_size:]

train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
test_dataset = full_test_dataset

print(f"New training set size: {len(train_dataset)}")
print(f"New validation set size: {len(val_dataset)}")
print(f"New test set size: {len(test_dataset)}")


# DataLoader configuration
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_size = len(train_loader)
val_size = len(val_loader)

print(f"Train size: {train_size} | Validation size: {val_size}")

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

# Training configuration
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimodal_model = multimodal_model.to(device)
multimodal_model.train()

# Optimizer
optimizer = schedulefree.AdamWScheduleFree(multimodal_model.parameters(), lr=3e-4)
optimizer.train()

# Scaler
scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
    training_losses = []
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = multimodal_model(batch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_losses.append(loss.item())

        # break early so we effectively don't train it
        if batch_idx > 1:
            break
    
    avg_train_loss = sum(training_losses) / len(training_losses)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    multimodal_model.eval()
    validation_losses = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = multimodal_model(batch)
            validation_losses.append(loss.item())

            # break early so we effectively don't train it
            if batch_idx > 1:
                break
        
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

            
    multimodal_model.train()
    os.makedirs(f"obscure_text_ocr_dummy_{epoch+1}", exist_ok=True)
    multimodal_model._save_model(f"obscure_text_ocr_dummy_{epoch+1}")