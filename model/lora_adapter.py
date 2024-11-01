import torch
import torch.nn as nn

class LoRAAdapter(nn.Module):
    def __init__(self, model, target_layers, r=4):
        super(LoRAAdapter, self).__init__()
        self.model = model
        self.r = r
        self.target_layers = target_layers
        self.lora_layers = self.initialize_lora_layers()

    def initialize_lora_layers(self):
        lora_layers = nn.ModuleDict()
        for name, layer in self.model.named_modules():
            if any(t in name for t in self.target_layers):
                lora_layer = nn.Linear(layer.in_features, self.r)
                lora_layers[name] = lora_layer
        return lora_layers

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        for name, layer in self.model.named_modules():
            if name in self.lora_layers:
                lora_output = self.lora_layers[name](output.last_hidden_state)
                output.last_hidden_state += lora_output
        return output
