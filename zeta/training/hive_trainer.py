"""
Train several models all at once.

HiveTrainer
we could make a framework that trains multiple LLMs all at once on separate threads and they fight for compute
time. This would be a good way to train a lot of models at once. We could also make it so that the models can
communicate with each other and share information

Features:
    * Train several models all at once
    * Models can communicate with each other and share information
    * Models fight for compute time, so the best model wins
    * Model with highest loss is killed and replaced with a new model
    * Models can be trained on separate GPUs
    * Multiple ModelS can be trained on a single GPU


"""

import torch
import torch.distributed as dist
import threading
from zeta.training.train import Trainer


def model_training_thread(model, *args, **kwargs):
    """Model training thread with model and args and kwargs"""
    trainer = Trainer(model=model, *args, **kwargs)
    trainer.train()


class HiveTrainer:
    """
    HiveTrainer

    Train several models all at once.

    HiveTrainer
    we could make a framework that trains multiple LLMs all at once on separate threads and they fight for compute
    time. This would be a good way to train a lot of models at once. We could also make it so that the models can
    communicate with each other and share information

    Features:
        * Train several models all at once
        * Models can communicate with each other and share information
        * Models fight for compute time, so the best model wins
        * Model with highest loss is killed and replaced with a new model
        * Models can be trained on separate GPUs
        * Multiple ModelS can be trained on a single GPU

    Args:
        models (list): List of models to train
        gradient_accumluate_every (int): Gradient accumulate every
        batch_size (int): Batch size
        seq_len (int): Sequence length
        entity_name (str): Entity name
        use_fsdp (bool): Use FSDP
        use_activation_checkpointing (bool): Use activation checkpointing
        learning_rate (float): Learning rate
        seed (int): Seed
        use_pretokenized (bool): Use pretokenized
        resume_from_checkpoint (str): Resume from checkpoint
        checkpointing_steps (int): Checkpointing steps
        output_dir (str): Output directory
        weight_decay (float): Weight decay
        use_deepspeed (bool): Use deepspeed

    Methods:
        train: Train the models

    Usage:
        # Instantiate models
        models = [YourModelClass1(), YourModelClass2()]  # Replace with your model classes





    """

    def __init__(
        self,
        models,
        gradient_accumluate_every,
        batch_size,
        seq_len,
        entity_name,
        use_fsdp=False,
        use_activation_checkpointing=False,
        learning_rate=None,
        seed=None,
        use_pretokenized=False,
        resume_from_checkpoint=None,
        checkpointing_steps=None,
        output_dir=None,
        weight_decay=None,
        use_deepspeed=None,
    ):
        self.models = models
        self.gradient_accumluate_every = gradient_accumluate_every
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.entity_name = entity_name
        self.use_fsdp = use_fsdp
        self.use_activation_checkpointing = use_activation_checkpointing
        self.learning_rate = learning_rate
        self.seed = seed
        self.use_pretokenized = use_pretokenized
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpointing_steps = checkpointing_steps
        self.output_dir = output_dir
        self.weight_decay = weight_decay
        self.use_deepspeed = use_deepspeed

    def train(
        self,
        MASTER_ADDR=None,
        MASTER_PORT=None,
        WORLD_SIZE=None,
        RANK=None,
    ):
        """
        Train the models

        Args:
            MASTER_ADDR (str): Master address
            MASTER_PORT (str): Master port
            WORLD_SIZE (str): World size
            RANK (str): Rank

        Usage:
            hive_trainer.train(MASTER_ADDR='localhost', MASTER_PORT='9994', RANK='0', WORLD_SIZE=str(torch.cuda.device_count()))

        """
        threads = []

        for model in self.models:
            t = threading.Thread(
                target=model_training_thread,
                args=(model,),
                kwargs={
                    "gradient_accumulate_every": self.gradient_accumulate_every,
                    "batch_size": self.batch_size,
                    "seq_len": self.seq_len,
                    "entity_name": self.entity_name,
                    "use_fsdp": self.use_fsdp,
                    "use_activation_checkpointing": (
                        self.use_activation_checkpointing
                    ),
                    "learning_rate": self.learning_rate,
                    "seed": self.seed,
                    "use_pretokenized": self.use_pretokenized,
                    "resume_from_checkpoint": self.resume_from_checkpoint,
                    "checkpointing_steps": self.checkpointing_steps,
                    "output_dir": self.output_dir,
                    "weight_decay": self.weight_decay,
                    "use_deepspeed": self.use_deepspeed,
                    "MASTER_ADDR": MASTER_ADDR,
                    "MASTER_PORT": MASTER_PORT,
                    "RANK": RANK,
                    "WORLD_SIZE": WORLD_SIZE,
                },
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


# # Instantiate models
# models = [YourModelClass1(), YourModelClass2()]  # Replace with your model classes

# # Instantiate HiveTrainer and begin training
# hive_trainer = HiveTrainer(
#     models=models,
#     gradient_accumulate_every=1,
#     batch_size=32,
#     seq_len=512,
#     entity_name="my_project",
#     learning_rate=0.001,
#     seed=42
# )

# hive_trainer.train(MASTER_ADDR='localhost', MASTER_PORT='9994', RANK='0', WORLD_SIZE=str(torch.cuda.device_count()))
