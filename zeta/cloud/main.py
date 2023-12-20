import logging
from typing import Any

from sky import AWS, Resources

from zeta.cloud.sky_api import SkyInterface

skyapi = SkyInterface(stream_logs_enabled=True)


# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def zetacloud(
    task_name: str = None,
    cluster_name: str = "ZetaTrainingRun",
    setup: str = "pip install -r requirements.txt",
    cloud: Any = AWS(),
    gpus: str = "V100:4",
    filename: str = "train.py",
    stop: bool = False,
    down: bool = False,
    status_report: bool = False,
    *args,
    **kwargs,
):
    """zetacloud

    Args:
        task_name (str, optional): _description_. Defaults to None.
        cluster_name (str, optional): _description_. Defaults to "[ZetaTrainingRun]".
        cloud (Any, optional): _description_. Defaults to AWS().
        gpus (str, optional): _description_. Defaults to None.
    """
    try:
        task = skyapi.create_task(
            name=task_name,
            setup=setup,
            run=f"python {filename}",
            workdir=".",
        )
        logger.info(f"Task: {task} has been created")

        # Set the resources
        task.set_resources(Resources(accelerators=gpus))
        # logger.info(f"Resources: {task.resources} have been set")

        # Execute the task on the cluster
        execution = skyapi.launch(task, cluster_name)
        print(execution)
        logger.info(
            f"Task: {task} has been launched on cluster: {cluster_name}"
        )

        if stop:
            skyapi.stop(cluster_name)
            logger.info(f"Cluster: {cluster_name} has been stopped")

        if down:
            skyapi.down(cluster_name)
            logger.info(f"Cluster: {cluster_name} has been deleted")

        if status_report:
            skyapi.status(cluster_names=[cluster_name])
            logger.info(f"Cluster: {cluster_name} has been reported on")

    except Exception as error:
        print(
            f"There has been an error: {error} the root cause is:"
            f" {error.__cause__}"
        )
