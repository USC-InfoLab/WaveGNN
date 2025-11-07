import argparse
import json


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="P12",
        choices=["P12", "P19", "MIMIC3-PHE", "MIMIC3-IHM", "PAM"],
        help="Select the dataset to use.",
    )

    args, _ = parser.parse_known_args()

    # determine the configuration file based on the dataset
    if args.dataset == "P12":
        config_file = "config_p12.json"
    elif args.dataset == "P19":
        config_file = "config_p19.json"
    elif args.dataset == "MIMIC3-PHE":
        config_file = "config_mimic3_phe.json"
    elif args.dataset == "MIMIC3-IHM":
        config_file = "config_mimic3_ihm.json"
    else:
        config_file = "config_PAM.json"

    # load the JSON config file
    with open(config_file, "r") as file:
        arg_vals = json.load(file)

    # add arguments from the JSON config file
    for key, value in arg_vals.items():
        parser.add_argument(
            f"--{key}", type=type(value), default=value, help=f"Description for {key}"
        )

    # parse all arguments, including those from the config file
    args = parser.parse_args()

    return args
