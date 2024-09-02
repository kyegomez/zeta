"""
Evolutionary Training for Transformer Models

This script implements an evolutionary algorithm to train transformer models
for text classification tasks. It uses a population of models, applies genetic
operations (crossover and mutation), and trains them over multiple generations.

Author: Claude
Date: 2024-09-02
"""

import random
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import os
import json
from datetime import datetime
from loguru import logger


def initialize_population(
    model_name: str, population_size: int, num_labels: int
) -> List[AutoModelForSequenceClassification]:
    """
    Initialize a population of transformer models.

    Args:
        model_name (str): The name of the pre-trained model to use as a base.
        population_size (int): The number of models in the population.
        num_labels (int): The number of labels for the classification task.

    Returns:
        List[AutoModelForSequenceClassification]: A list of initialized models.
    """
    logger.info(
        f"Initializing population with {population_size} models based on {model_name}"
    )
    population = [
        AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        for _ in range(population_size)
    ]
    logger.info(f"Population initialized with {len(population)} models")
    return population


def evaluate_model(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate a model's performance on a dataset.

    Args:
        model (AutoModelForSequenceClassification): The model to evaluate.
        dataloader (DataLoader): The DataLoader containing the evaluation data.
        device (torch.device): The device to run the evaluation on.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    logger.info(
        f"Evaluation - Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
    return avg_loss, accuracy


def crossover(
    parent1: AutoModelForSequenceClassification,
    parent2: AutoModelForSequenceClassification,
) -> AutoModelForSequenceClassification:
    """
    Perform crossover between two parent models to create a child model.

    Args:
        parent1 (AutoModelForSequenceClassification): The first parent model.
        parent2 (AutoModelForSequenceClassification): The second parent model.

    Returns:
        AutoModelForSequenceClassification: The child model resulting from crossover.
    """
    logger.debug("Performing crossover")
    child = AutoModelForSequenceClassification.from_pretrained(
        parent1.config._name_or_path, num_labels=parent1.config.num_labels
    )

    for name, param in child.named_parameters():
        if random.random() < 0.5:
            param.data.copy_(parent1.state_dict()[name])
        else:
            param.data.copy_(parent2.state_dict()[name])

    return child


def mutate(
    model: AutoModelForSequenceClassification, mutation_rate: float
) -> None:
    """
    Apply mutation to a model's parameters.

    Args:
        model (AutoModelForSequenceClassification): The model to mutate.
        mutation_rate (float): The probability of mutating each parameter.
    """
    logger.debug(f"Applying mutation with rate {mutation_rate}")
    for param in model.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param.data) * 0.1


def preprocess_function(examples, tokenizer, max_length=128):
    """
    Tokenize and pad the input texts, and prepare the labels.

    Args:
        examples (dict): The examples to preprocess.
        tokenizer (AutoTokenizer): The tokenizer to use.
        max_length (int): The maximum length for padding/truncation.

    Returns:
        dict: The preprocessed examples.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokenized["labels"] = examples["label"]
    return tokenized


def evolutionary_training(
    model_name: str,
    num_generations: int,
    population_size: int,
    mutation_rate: float,
    device: torch.device,
    dataset_name: str = "rotten_tomatoes",
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    max_length: int = 128,
) -> AutoModelForSequenceClassification:
    logger.info("Starting evolutionary training with the following parameters:")

    # Load dataset and tokenizer
    logger.info("Loading dataset and tokenizer...")
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Dataset and tokenizer loaded successfully.")

    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Preprocessing dataset",
    )
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    logger.info("Dataset preprocessing completed.")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=16, shuffle=True
    )
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=16)
    logger.info("Data loaders created successfully.")

    # Initialize population
    logger.info("Initializing population...")
    population = initialize_population(
        model_name, population_size, num_labels=2
    )
    logger.info(f"Population initialized with {len(population)} models.")

    best_accuracy = 0.0
    best_model = None

    for generation in range(num_generations):
        logger.info(f"Starting Generation {generation + 1}/{num_generations}")

        # Evaluate population
        logger.info("Evaluating population...")
        fitness_scores = []
        for i, model in enumerate(population):
            loss, accuracy = evaluate_model(model, eval_dataloader, device)
            fitness_scores.append((loss, accuracy))
            logger.info(
                f"  Model {i+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}"
            )

        # Select parents
        parents = sorted(
            zip(population, fitness_scores), key=lambda x: x[1][1], reverse=True
        )[:2]
        logger.info(
            f"Best two models selected with accuracies: {parents[0][1][1]:.4f}, {parents[1][1][1]:.4f}"
        )

        # Update best model if necessary
        if parents[0][1][1] > best_accuracy:
            best_accuracy = parents[0][1][1]
            best_model = parents[0][0]
            logger.info(
                f"New best model found with accuracy: {best_accuracy:.4f}"
            )

        # Create new population
        logger.info("Creating new population...")
        new_population = [
            parents[0][0],
            parents[1][0],
        ]  # Keep the two best models
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(parents, k=2)
            child = crossover(parent1[0], parent2[0])
            mutate(child, mutation_rate)
            new_population.append(child)
        logger.info(
            f"New population created with {len(new_population)} models."
        )

        population = new_population

        # Train the population
        logger.info("Training the population...")
        for i, model in enumerate(population):
            logger.info(f"Training model {i+1}/{len(population)}")
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train_dataloader) * num_epochs,
            )
            model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for batch in tqdm(
                    train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
                ):
                    optimizer.zero_grad()
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**inputs)
                    loss = outputs.loss
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                avg_loss = total_loss / len(train_dataloader)
                logger.info(
                    f"  Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}"
                )

        logger.info(f"Generation {generation + 1} completed.")

    # Final evaluation of the best model
    logger.info("Performing final evaluation of the best model...")
    best_model.to(device)
    final_loss, final_accuracy = evaluate_model(
        best_model, eval_dataloader, device
    )
    logger.info(
        f"Final evaluation - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}"
    )

    return best_model


def save_training_config(config: dict, output_dir: str) -> None:
    """
    Save the training configuration to a JSON file.

    Args:
        config (dict): The configuration dictionary to save.
        output_dir (str): The directory to save the configuration file in.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training configuration saved to {config_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Training configuration
    config = {
        "model_name": "distilbert-base-uncased",
        "dataset_name": "rotten_tomatoes",
        "num_generations": 5,
        "population_size": 4,
        "mutation_rate": 0.01,
        "num_epochs": 3,
        "learning_rate": 5e-5,
    }

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evolved_model_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save training configuration
    save_training_config(config, output_dir)

    # Perform evolutionary training
    best_model = evolutionary_training(
        model_name=config["model_name"],
        num_generations=config["num_generations"],
        population_size=config["population_size"],
        mutation_rate=config["mutation_rate"],
        device=device,
        dataset_name=config["dataset_name"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
    )

    # Save the best model
    model_save_path = os.path.join(output_dir, "best_model")
    best_model.save_pretrained(model_save_path)
    logger.info(f"Best model saved to {model_save_path}")

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer_save_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"Tokenizer saved to {tokenizer_save_path}")

    logger.info("Evolutionary training completed successfully")
