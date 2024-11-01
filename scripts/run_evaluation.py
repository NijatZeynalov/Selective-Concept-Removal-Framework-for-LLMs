
import argparse
import os
from training.evaluation import evaluate_model
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the concept removal model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to evaluation data")
    parser.add_argument('--concept_path', type=str, required=True, help="Path to concept definitions")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument('--base_model_path', type=str, required=True, help="Path to the base LLaMA model")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # Argument validation
    if not os.path.exists(args.data_path):
        logger.error(f"Evaluation data path {args.data_path} does not exist.")
        exit(1)
    if not os.path.exists(args.concept_path):
        logger.error(f"Concept definitions path {args.concept_path} does not exist.")
        exit(1)
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint path {args.model_path} does not exist.")
        exit(1)
    if not os.path.exists(args.base_model_path):
        logger.error(f"Base model path {args.base_model_path} does not exist.")
        exit(1)

    # Start evaluation
    logger.info("Starting evaluation process...")
    evaluate_model(args.data_path, args.concept_path, args.model_path, args.base_model_path)
    logger.info("Evaluation completed successfully.")
