import torch
from data.data_loader import get_data_loader
from utils.config import Config
from model.concept_removal_model import ConceptRemovalModel
from transformers import LlamaTokenizer
from sklearn.metrics import accuracy_score

def evaluate_model(data_path, concept_path, model_path, base_model_path):
    config = Config()
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    data_loader = get_data_loader(data_path, concept_path, tokenizer, config.batch_size, config.max_length)

    model = ConceptRemovalModel(base_model_path, config.target_layers, config.lora_rank).to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_accuracy = 0
    for input_ids, attention_mask in data_loader:
        input_ids, attention_mask = input_ids.to(config.device), attention_mask.to(config.device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        predictions = outputs.logits.argmax(dim=-1)
        accuracy = accuracy_score(input_ids.cpu().flatten(), predictions.cpu().flatten())
        total_accuracy += accuracy

    print(f"Average Accuracy: {total_accuracy/len(data_loader)}")
