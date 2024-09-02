import logging

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProteinDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 512):
        self.dataset = load_dataset("uniref50", split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(
            item["sequence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}


class ProteinGenerator:
    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_uniref50"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        logger.info(f"Loaded model: {model_name}")

    def prepare_dataset(self, batch_size: int = 8) -> DataLoader:
        """Prepare and process the dataset."""
        dataset = ProteinDataset(self.tokenizer)
        logger.info(f"Loaded UniRef50 dataset with {len(dataset)} sequences")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 5e-5,
    ) -> None:
        """Train the model."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(dataloader)
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}"
            )

    def generate_protein(self, prompt: str, max_length: int = 512) -> str:
        """Generate a protein sequence based on a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # Initialize the protein generator
    protein_gen = ProteinGenerator()

    # Prepare the dataset
    dataloader = protein_gen.prepare_dataset()

    # Train the model
    protein_gen.train(dataloader)

    # Example inference
    prompts = [
        "Generate a protein sequence that targets COVID-19 spike protein:",
        "Design an enzyme that breaks down plastic:",
        "Create an antibody that binds to cancer cells:",
    ]

    for prompt in prompts:
        generated_sequence = protein_gen.generate_protein(prompt)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated protein sequence: {generated_sequence}\n")


if __name__ == "__main__":
    main()

#####