from typing import List

import sky
from sky import Task


class SkyInterface:
    """

    SkyInterface is a wrapper around the sky Python API. It provides a
    simplified interface for launching, executing, stopping, starting, and
    tearing down clusters.

    Attributes:
        clusters (dict): A dictionary of clusters that have been launched.
        The keys are the names of the clusters and the values are the handles
        to the clusters.

    Methods:
        launch: Launch a cluster
        execute: Execute a task on a cluster
        stop: Stop a cluster
        start: Start a cluster
        down: Tear down a cluster
        status: Get the status of a cluster
        autostop: Set the autostop of a cluster

    Example:
        >>> sky_interface = SkyInterface()
        >>> job_id = sky_interface.launch("task", "cluster_name")
        >>> sky_interface.execute("task", "cluster_name")
        >>> sky_interface.stop("cluster_name")
        >>> sky_interface.start("cluster_name")
        >>> sky_interface.down("cluster_name")
        >>> sky_interface.status()
        >>> sky_interface.autostop("cluster_name")


    """

    def __init__(
        self,
        task_name: str = None,
        cluster_name: str = None,
        gpus: str = "T4:1",
        stream_logs_enabled: bool = False,
        *args,
        **kwargs,
    ):
        self.task_name = task_name
        self.cluster_name = cluster_name
        self.gpus = gpus
        self.stream_logs_enabled = stream_logs_enabled
        self.clusters = {}

    def launch(self, task: Task = None, cluster_name: str = None, **kwargs):
        """Launch a task on a cluster

        Args:
            task (str): code to execute on the cluster
            cluster_name (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        cluster = None
        try:
            cluster = sky.launch(
                task=task,
                cluster_name=cluster_name,
                stream_logs=self.stream_logs_enabled,
                **kwargs,
            )
            print(f"Launched job {cluster} on cluster {cluster_name}")
            return cluster
        except Exception as error:
            # Deep error logging
            print(
                f"Error launching job {cluster} on cluster {cluster_name} with"
                f" error {error}"
            )
            raise error

    def execute(self, task: Task = None, cluster_name: str = None, **kwargs):
        """Execute a task on a cluster

        Args:
            task (_type_): _description_
            cluster_name (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if cluster_name not in self.clusters:
            raise ValueError("Cluster {} does not exist".format(cluster_name))
        try:
            return sky.exec(
                task=task,
                cluster_name=cluster_name,
                stream_logs=self.stream_logs_enabled,
                **kwargs,
            )
        except Exception as e:
            print("Error executing on cluster:", e)

    def stop(self, cluster_name: str = None, **kwargs):
        """Stop a cluster

        Args:
            cluster_name (str): name of the cluster to stop
        """
        try:
            sky.stop(cluster_name, **kwargs)
        except (ValueError, RuntimeError) as e:
            print("Error stopping cluster:", e)

    def start(self, cluster_name: str = None, **kwargs):
        """start a cluster

        Args:
            cluster_name (str): name of the cluster to start
        """
        try:
            sky.start(cluster_name, **kwargs)
        except Exception as e:
            print("Error starting cluster:", e)

    def down(self, cluster_name: str = None, **kwargs):
        """Down a cluster

        Args:
            cluster_name (str): name of the cluster to tear down
        """
        try:
            sky.down(cluster_name, **kwargs)
            if cluster_name in self.clusters:
                del self.clusters[cluster_name]
        except (ValueError, RuntimeError) as e:
            print("Error tearing down cluster:", e)

    def status(self, cluster_names: List[str] = None, **kwargs):
        """Save a cluster

        Returns:
            r: the status of the cluster
        """
        try:
            return sky.status(cluster_names, **kwargs)
        except Exception as e:
            print("Error getting status:", e)

    def autostop(self, cluster_name: str = None, **kwargs):
        """Autostop a cluster

        Args:
            cluster_name (str): name of the cluster to autostop
        """
        try:
            sky.autostop(cluster_name, **kwargs)
        except Exception as e:
            print("Error setting autostop:", e)

    def create_task(
        self,
        name: str = None,
        setup: str = None,
        run: str = None,
        workdir: str = None,
        task: str = None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to None.
            setup (str, optional): _description_. Defaults to None.
            run (str, optional): _description_. Defaults to None.
            workdir (str, optional): _description_. Defaults to None.
            task (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_

        # A Task that will sync up local workdir '.', containing
        # requirements.txt and train.py.
        sky.Task(setup='pip install requirements.txt',
                run='python train.py',
                workdir='.')

        # An empty Task for provisioning a cluster.
        task = sky.Task(num_nodes=n).set_resources(...)

        # Chaining setters.
        sky.Task().set_resources(...).set_file_mounts(...)
        """
        return Task(
            name=name, setup=setup, run=run, workdir=workdir, *args, **kwargs
        )
