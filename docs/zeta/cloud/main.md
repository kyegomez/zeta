
# ZetaCloud Documentation

## Overview

ZetaCloud is a versatile command-line tool that simplifies the process of training or fine-tuning machine learning models on remote GPU clusters. With just a few commands, you can effortlessly manage your tasks and harness the computational power of various GPUs. This comprehensive documentation will guide you through every aspect of the ZetaCloud CLI, from installation to advanced usage.

## Table of Contents

1. [Installation](#installation)
2. [ZetaCloud CLI](#zetacloud-cli)
   - [Options](#options)
3. [Basic Usage](#basic-usage)
   - [Example 1: Starting a Task](#example-1-starting-a-task)
   - [Example 2: Stopping a Task](#example-2-stopping-a-task)
   - [Example 3: Checking Task Status](#example-3-checking-task-status)
4. [Advanced Usage](#advanced-usage)
   - [Example 4: Cluster Selection](#example-4-cluster-selection)
   - [Example 5: Choosing the Cloud Provider](#example-5-choosing-the-cloud-provider)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Installation <a name="installation"></a>

Getting started with ZetaCloud is quick and straightforward. Follow these steps to set up ZetaCloud on your machine:

1. Open your terminal or command prompt.

2. Install the `zetascale` package using `pip`:

   ```bash
   pip install zetascale
   ```

3. After a successful installation, you can access the ZetaCloud CLI by running the following command:

   ```bash
   zeta -h
   ```

   This command will display a list of available options and basic usage information for ZetaCloud.

## 2. ZetaCloud CLI <a name="zetacloud-cli"></a>

The ZetaCloud Command-Line Interface (CLI) provides a set of powerful options that enable you to manage tasks on GPU clusters effortlessly. Below are the available options:

### Options <a name="options"></a>

- `-h, --help`: Display the help message and exit.
- `-t TASK_NAME, --task_name TASK_NAME`: Specify the name of your task.
- `-c CLUSTER_NAME, --cluster_name CLUSTER_NAME`: Specify the name of the cluster you want to use.
- `-cl CLOUD, --cloud CLOUD`: Choose the cloud provider (e.g., AWS, Google Cloud, Azure).
- `-g GPUS, --gpus GPUS`: Specify the number and type of GPUs required for your task.
- `-f FILENAME, --filename FILENAME`: Provide the filename of your Python script or code.
- `-s, --stop`: Use this flag to stop a running task.
- `-d, --down`: Use this flag to terminate a cluster.
- `-sr, --status_report`: Check the status of your task.

## 3. Basic Usage <a name="basic-usage"></a>

ZetaCloud's basic usage covers essential tasks such as starting, stopping, and checking the status of your tasks. Let's explore these tasks with examples.

### Example 1: Starting a Task <a name="example-1-starting-a-task"></a>

To start a task, you need to specify the Python script you want to run and the GPU configuration. Here's an example command:

```bash
zeta -f train.py -g A100:8
```

In this example:
- `-f train.py` indicates that you want to run the Python script named `train.py`.
- `-g A100:8` specifies that you require 8 NVIDIA A100 GPUs for your task.

### Example 2: Stopping a Task <a name="example-2-stopping-a-task"></a>

If you need to stop a running task, you can use the following command:

```bash
zeta -s
```

This command will stop the currently running task.

### Example 3: Checking Task Status <a name="example-3-checking-task-status"></a>

To check the status of your task, use the following command:

```bash
zeta -sr
```

This command will provide you with a detailed status report for your active task.

## 4. Advanced Usage <a name="advanced-usage"></a>

ZetaCloud also offers advanced options that allow you to fine-tune your tasks according to your specific requirements.

### Example 4: Cluster Selection <a name="example-4-cluster-selection"></a>

You can select a specific cluster for your task by providing the cluster name with the `-c` option:

```bash
zeta -f train.py -g A100:8 -c my_cluster
```

This command will run your task on the cluster named `my_cluster`.

### Example 5: Choosing the Cloud Provider <a name="example-5-choosing-the-cloud-provider"></a>

ZetaCloud supports multiple cloud providers. You can specify your preferred cloud provider using the `-cl` option:

```bash
zeta -f train.py -g A100:8 -cl AWS
```

This command will execute your task on a cloud provider's infrastructure, such as AWS.

## 5. Additional Information <a name="additional-information"></a>

- ZetaCloud simplifies the process of utilizing GPU clusters, allowing you to focus on your machine learning tasks rather than infrastructure management.

- You can easily adapt ZetaCloud to various cloud providers, making it a versatile tool for your machine learning needs.

