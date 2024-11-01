import torch

class Config:
    def __init__(self):
        self.batch_size = 8
        self.epochs = 3
        self.learning_rate = 5e-5
        self.max_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_layers = ["ffn", "attn"]
        self.lora_rank = 4
