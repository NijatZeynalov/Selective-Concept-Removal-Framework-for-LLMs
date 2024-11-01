import torch
import torch.nn as nn
from transformers import LlamaForCausalLM


class ConceptRemovalModel(nn.Module):
    def __init__(self, base_model_path, target_layers, lora_rank=4):
        super(ConceptRemovalModel, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(base_model_path)
        self.adapter = LoRAAdapter(self.model, target_layers, r=lora_rank)

    def forward(self, input_ids, attention_mask):
        return self.adapter(input_ids, attention_mask)

    def remove_concept(self, concept_embeddings):
        for name, layer in self.adapter.lora_layers.items():
            with torch.no_grad():
                layer.weight -= concept_embeddings
