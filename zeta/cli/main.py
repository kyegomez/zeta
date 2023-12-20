import argparse
from zeta.cloud.main import zetacloud


def main():
    """Main function for the CLI

    Args:
        task_name (str, optional): _description_. Defaults to None.
        cluster_name (str, optional): _description_. Defaults to "[ZetaTrainingRun]".
        cloud (Any, optional): _description_. Defaults to AWS().
        gpus (str, optional): _description_. Defaults to None.

    Examples:
        $ zetacloud -t "test" -c "[ZetaTrainingRun]" -cl AWS -g "1 V100"


    """
    parser = argparse.ArgumentParser(description="Zetacloud CLI")
    parser.add_argument("-t", "--task_name", type=str, help="Task name")
    parser.add_argument(
        "-c",
        "--cluster_name",
        type=str,
        default="[ZetaTrainingRun]",
        help="Cluster name",
    )
    parser.add_argument(
        "-cl", "--cloud", type=str, default="AWS", help="Cloud provider"
    )
    parser.add_argument("-g", "--gpus", type=str, help="GPUs")
    parser.add_argument(
        "-f", "--filename", type=str, default="train.py", help="Filename"
    )
    parser.add_argument("-s", "--stop", action="store_true", help="Stop flag")
    parser.add_argument("-d", "--down", action="store_true", help="Down flag")
    parser.add_argument(
        "-sr", "--status_report", action="store_true", help="Status report flag"
    )

    # Generate API key
    # parser.add_argument(
    #     "-k", "--generate_api_key", action="store_true", help="Generate key flag"
    # )

    # Sign In
    # parser.add_argument(
    #     "-si", "--sign_in", action="store_true", help="Sign in flag"
    # )

    args = parser.parse_args()

    zetacloud(
        task_name=args.task_name,
        cluster_name=args.cluster_name,
        cloud=args.cloud,
        gpus=args.gpus,
        filename=args.filename,
        stop=args.stop,
        down=args.down,
        status_report=args.status_report,
    )


# if __name__ == "__main__":
#     main()
