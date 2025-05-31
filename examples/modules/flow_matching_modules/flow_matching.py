from pathlib import Path
from loguru import logger
from zeta import Flow, FlowConfig


if __name__ == "__main__":
    # Setup logging
    logger.add("flow_model.log", rotation="500 MB")

    # Initialize and train model
    config = FlowConfig()
    flow = Flow(config)
    losses = flow.train_model(save_dir=Path("checkpoints"))

    # Visualize results
    flow.visualize_flow(save_path=Path("flow_visualization.png"))
