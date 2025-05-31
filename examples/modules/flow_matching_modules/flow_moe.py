from pathlib import Path

from loguru import logger
from zeta.nn.modules.flow_matching import MixtureFlow, MixtureFlowConfig

if __name__ == "__main__":
    # Setup configuration
    config = MixtureFlowConfig()

    # Initialize and train model
    model = MixtureFlow(config)
    training_metrics = model.train_model(save_dir=Path("checkpoints"))

    # Visualize results
    model.visualize_flow(save_path=Path("visualizations/flow_evolution.png"))
    model.visualize_expert_specialization(
        save_path=Path("visualizations/expert_specialization.png")
    )

    # Get and log expert statistics
    expert_stats = model.get_expert_statistics()
    logger.info("\nExpert Statistics:")
    for stat_name, values in expert_stats.items():
        logger.info(f"\n{stat_name}:")
        for expert_id, value in enumerate(values):
            logger.info(f"Expert {expert_id}: {value:.3f}")
