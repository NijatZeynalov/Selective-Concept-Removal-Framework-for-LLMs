import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.data_loader import get_data_loader
from utils.config import Config
from model.concept_removal_model import ConceptRemovalModel
from transformers import LlamaTokenizer


def train_model(data_path, concept_path, base_model_path):
    config = Config()
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    data_loader = get_data_loader(data_path, concept_path, tokenizer, config.batch_size, config.max_length)

    model = ConceptRemovalModel(base_model_path, config.target_layers, config.lora_rank).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0
        for input_ids, attention_mask in tqdm(data_loader):
            input_ids, attention_mask = input_ids.to(config.device), attention_mask.to(config.device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {epoch_loss / len(data_loader)}")
    torch.save(model.state_dict(), "model_checkpoint.pt")
