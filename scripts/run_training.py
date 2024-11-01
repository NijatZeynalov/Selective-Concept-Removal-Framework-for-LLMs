
import argparse
import os
from training.train import train_model
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the concept removal model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to training data")
    parser.add_argument('--concept_path', type=str, required=True, help="Path to concept definitions")
    parser.add_argument('--base_model_path', type=str, required=True, help="Path to the base LLaMA model")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # Argument validation
    if not os.path.exists(args.data_path):
        logger.error(f"Training data path {args.data_path} does not exist.")
        exit(1)
    if not os.path.exists(args.concept_path):
        logger.error(f"Concept definitions path {args.concept_path} does not exist.")
        exit(1)
    if not os.path.exists(args.base_model_path):
        logger.error(f"Base model path {args.base_model_path} does not exist.")
        exit(1)

    # Start training
    logger.info("Starting training process...")
    train_model(args.data_path, args.concept_path, args.base_model_path)
    logger.info("Training completed successfully.")
