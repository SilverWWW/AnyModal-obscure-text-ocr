import llm
import anymodal
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='GET_YOUR_OWN_TOKEN_FROM_HUGGINGFACE',   
    use_peft=False
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Dataset configuration
dataset_name = "linxy/LaTeX_OCR"

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)
# image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('wanglab/medsam-vit-base', use_peft=False)
# image_processor, vision_model, vision_hidden_size = vision.get_image_encoder("flaviagiammarino/pubmed-clip-vit-base-patch32", use_peft=False)
# image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('emre570/google-vit-large-finetuned', use_peft=True)


# Load dataset
train_dataset = vision.ImageDataset(dataset_name, image_processor, name = 'human_handwrite', split = 'train')
val_dataset = vision.ImageDataset(dataset_name, image_processor, name = 'human_handwrite', split = 'validation')
train_size = len(train_dataset)
val_size = len(val_dataset)

print(f"Train size: {train_size} | Validation size: {val_size}")

# DataLoader configuration
batch_size = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    prompt_text="The latex expression of the equation in the image is: ")

# Training configuration
num_epochs = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimodal_model = multimodal_model.to(device)
multimodal_model.train()

# Optimizer
optimizer = schedulefree.AdamWScheduleFree(multimodal_model.parameters(), lr=3e-4)
optimizer.train()

os.makedirs("latex_ocr", exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    training_losses = []
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        logits, loss = multimodal_model(batch)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
    
    avg_train_loss = sum(training_losses) / len(training_losses)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    multimodal_model.eval()
    validation_losses = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False):
            logits, loss = multimodal_model(batch)
            validation_losses.append(loss.item())
        
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # Decode a random validation sample
        for _ in range(5):
            sample_idx = np.random.randint(len(val_dataset))
            sample = val_dataset[sample_idx]
            print("Actual LaTeX: ", sample['text'])
            print("Generated LaTeX: ", multimodal_model.generate(sample['input'], max_new_tokens=120))
            
    multimodal_model.train()
    multimodal_model._save_model('latex_ocr')


# evaluate on test set
test_dataset = vision.ImageDataset(dataset_name, image_processor, name = 'human_handwrite_print', split = 'test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

multimodal_model.eval()
test_losses = []

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(test_loader), desc=f"Test", leave=False):
        logits, loss = multimodal_model(batch)
        test_losses.append(loss.item())
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Decode a random test sample
    for _ in range(5):
        sample_idx = np.random.randint(len(test_dataset))
        sample = test_dataset[sample_idx]
        print("Actual LaTeX: ", sample['text'])
        print("Generated LaTeX: ", multimodal_model.generate(sample['input'], max_new_tokens=120))
