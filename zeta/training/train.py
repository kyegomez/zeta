import math
import os
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator, set_seed

from zeta.optim.decoupled_optimizer import decoupled_optimizer
from zeta.training.activation_checkpoint import activation_checkpointing
from zeta.training.dataloader import build_dataloaders, build_pre_tokenized
from zeta.training.fsdp import fsdp
from zeta.training.scheduler import get_lr_scheduler_with_warmup


def print_num_params(model, accelerator: Accelerator):
    """Print number of parameters in model"""
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")


def Trainer(
    gradient_accumulate_every: int = 2,
    batch_size: int = None,
    seq_len: int = None,
    entity_name: str = "zeta",
    model=None,
    use_fsdp: bool = False,
    use_activation_checkpointing: bool = False,
    learning_rate: float = None,
    seed: int = None,
    use_pretokenized: bool = False,
    resume_from_checkpoint: bool = None,
    checkpointing_steps=None,
    output_dir: str = "checlpoints/",
    optimizer_type: str = "Adam8bit",
    weight_decay: float = 0.1,
    use_deepspeed=None,
    *args,
    **kwargs,
):
    """Trainer

    Args:
        gradient_accumulate_every (int, optional): _description_. Defaults to None.
        batch_size (int, optional): _description_. Defaults to None.
        seq_len (int, optional): _description_. Defaults to None.
        entity_name (str, optional): _description_. Defaults to None.
        model (_type_, optional): _description_. Defaults to None.
        use_fsdp (bool, optional): _description_. Defaults to False.
        use_activation_checkpointing (bool, optional): _description_. Defaults to False.
        learning_rate (_type_, optional): _description_. Defaults to None.
        seed (_type_, optional): _description_. Defaults to None.
        use_pretokenized (bool, optional): _description_. Defaults to False.
        resume_from_checkpoint (_type_, optional): _description_. Defaults to None.
        checkpointing_steps (_type_, optional): _description_. Defaults to None.
        output_dir (_type_, optional): _description_. Defaults to None.
        weight_decay (_type_, optional): _description_. Defaults to None.
        use_deepspeed (_type_, optional): _description_. Defaults to None.

    Examples:
    >>> Trainer(
    >>>     gradient_accumulate_every=gradient_accumulate_every,
    >>>     batch_size=batch_size,
    >>>     seq_len=seq_len,
    >>>     entity_name=entity_name,
    >>>     model=model,
    >>>     use_fsdp=use_fsdp,
    >>>     use_activation_checkpointing=use_activation_checkpointing,
    >>>     learning_rate=learning_rate,
    >>>     seed=seed,
    >>>     use_pretokenized=use_pretokenized,
    >>>     resume_from_checkpoint=resume_from_checkpoint,
    >>>     checkpointing_steps=checkpointing_steps,
    >>>     output_dir=output_dir,
    >>>     weight_decay=weight_decay,
    >>>     use_deepspeed=use_deepspeed,
    >>> )

    """
    # accelerator

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulate_every,
        mixed_precision="fp16",
        log_with="wandb",
        kwargs_handlers=[timeout],
    )
    # AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_

    accelerator.init_trackers(
        project_name=entity_name,
        config={
            "batch_size": batch_size,
            "gradient_accumulate_every": gradient_accumulate_every,
            "learning_rate": learning_rate,
            "seq_len": seq_len,
        },
        init_kwargs={"wandb": {"entity": entity_name}},
    )

    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # set seed

    set_seed(seed)

    model = model().to(accelerator.device)

    print_num_params(model, accelerator)

    if use_fsdp:
        model = fsdp(model, mp="fp16", shard_strat="SHARD_GRAD")

    if use_activation_checkpointing:
        activation_checkpointing(model, accelerator)

    model = accelerator.prepare(model)

    # dataloaders

    if use_pretokenized:
        train_dataset = build_pre_tokenized()
    else:
        train_dataset = build_dataloaders()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    # optimizer

    optim = decoupled_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta_1=0.90,
        beta_2=0.95,
        optimizer_type=optimizer_type,
        use_fsdp=True,
        accelerator=accelerator,
    )

    # Determine number of training steps

    max_train_steps = math.ceil(len(train_loader) / gradient_accumulate_every)
    accelerator.print(f"Max train steps: {max_train_steps}")

    # lr scheduler

    NUM_WARMUP_STEPS = int(max_train_steps * 0.01)
    accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

    lr_scheduler = get_lr_scheduler_with_warmup(
        optimizer=optim,
        scheduler_type="cosine",
        num_warmup_steps=NUM_WARMUP_STEPS,
        max_train_steps=max_train_steps,
        grad_accumulate_every=gradient_accumulate_every,
    )

    # prepare

    optim, train_loader, lr_scheduler = accelerator.prepare(
        optim, train_loader, lr_scheduler
    )

    # checkpoint scheduler

    accelerator.register_for_checkpointing(lr_scheduler)

    # I do not know why Huggingface recommends recalculation of max_train_steps

    max_train_steps = math.ceil(len(train_loader) / gradient_accumulate_every)
    accelerator.print(f"Max train steps recalculated: {max_train_steps}")

    # Total batch size for logging

    total_batch_size = (
        batch_size * accelerator.num_processes * gradient_accumulate_every
    )
    accelerator.print(f"Total batch size: {total_batch_size}")

    # resume training

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if resume_from_checkpoint:
        if resume_from_checkpoint is not None or resume_from_checkpoint != "":
            accelerator.print(
                f"Resuming from checkpoint {resume_from_checkpoint}"
            )
            accelerator.load_state(resume_from_checkpoint)
            path = os.path.basename(resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = (
            int(training_difference.replace("step_", ""))
            * gradient_accumulate_every
        )

    if resume_from_checkpoint and resume_step is not None:
        train_loader = accelerator.skip_first_batches(train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)

    # training

    model.train()
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            inputs = batch["input_ids"].to(accelerator.device)
            loss = model(inputs, return_loss=True)
            accelerator.backward(loss)

            accelerator.log({"loss": loss.item()}, step=step)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            lr_scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps }"
                if output_dir is not None:
                    output_dir = os.path.join(output_dir, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= max_train_steps:
            break

    # end training

    accelerator.print("Training Finished")
    accelerator.end_training()

    # save final model

    accelerator.print(f"Saving model to {output_dir}")
    if output_dir is not None:
        accelerator.wait_for_everyone()
        unwrappedim = accelerator.unwrap_model(model)
        with accelerator.main_process_first():
            accelerator.save(
                unwrappedim.state_dict(),
                f"{output_dir}/final/final_model.pt",
            )


def train(
    MASTER_ADDR=None,
    MASTER_PORT=None,
    RANK=None,
    WORLD_SIZE=None,
    *args,
    **kwargs,
):
    os.environ["MASTER_ADDR"] or MASTER_ADDR  # = 'localhost'
    os.environ["MASTER_PORT"] or MASTER_PORT  # = '9994'

    # # [CRITICAL] Pay attention to this when scaling to multiple GPUs and clusters

    os.environ["RANK"] or RANK  # = str(0) # Number of nodes (servers)
    os.environ["WORLD_SIZE"] or WORLD_SIZE  # = str(torch.cuda.device_count())

    torch.distributed.init_process_group()

    Trainer(*args, **kwargs)
