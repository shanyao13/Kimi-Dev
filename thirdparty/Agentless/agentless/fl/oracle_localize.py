import argparse
import json
import os

import unidiff
from datasets import load_dataset
from tqdm import tqdm

from agentless.util.utils import setup_logger


def get_oracle_filenames(instance):
    """
    Returns the filenames that are changed in the patch
    """
    if not instance.get("patch"):
        return set()
    source_files = {
        patch_file.source_file.split("a/", 1)[-1]
        for patch_file in unidiff.PatchSet(instance["patch"])
        if patch_file.source_file.startswith("a/")
    }
    gold_docs = set()
    for source_file in source_files:
        gold_docs.add(source_file)
    return gold_docs


def oracle_localize(args):
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        with open(args.dataset, "r") as f:
            swe_bench_data = [json.loads(line) for line in f]
    else:
        swe_bench_data = load_dataset(args.dataset, split="test")

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)

    output_file = os.path.join(args.output_folder, args.output_file)

    if args.target_id:
        swe_bench_data = [
            x for x in swe_bench_data if x["instance_id"] == args.target_id
        ]

    for bug in tqdm(swe_bench_data, colour="MAGENTA"):
        instance_id = bug["instance_id"]
        log_file = os.path.join(
            args.output_folder, "localization_logs", f"{instance_id}.log"
        )
        logger = setup_logger(log_file)
        logger.info(f"Processing bug {instance_id}")

        oracle_files = sorted(list(get_oracle_filenames(bug)))
        if args.skip_non_python:
            oracle_files = [f for f in oracle_files if f.endswith(".py")]

        result = {
            "instance_id": instance_id,
            "found_files": oracle_files,
            "additional_artifact_loc_file": None,
            "file_traj": {},
            "found_related_locs": {},
            "additional_artifact_loc_related": None,
            "related_loc_traj": [],
            "found_edit_locs": {},
            "additional_artifact_loc_edit_location": None,
            "edit_loc_traj": {},
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="Current supported dataset for evaluation or path to a local json/jsonl file",
    )
    parser.add_argument("--target_id", type=str, help="Target instance ID")
    parser.add_argument(
        "--skip_non_python", action="store_true", help="Skip non-python files"
    )

    args = parser.parse_args()

    oracle_localize(args)


if __name__ == "__main__":
    main()
