import os
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd

from clip import evaluate_images

IMAGES_PATH = Path("images")

VERSIONS = ["1.1", "1.5", "2.0", "xl_1-0"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def submit_job(script_filename: str) -> str:
    result = subprocess.run(
        ["sbatch", script_filename], stdout=subprocess.PIPE, text=True
    )
    output = result.stdout
    job_id = output.strip().split()[-1]  # Extract job ID from the output
    return job_id


def check_job_status(job_id: str) -> str:
    result = subprocess.run(
        ["squeue", "--job", job_id, "--noheader"], stdout=subprocess.PIPE, text=True
    )
    output = result.stdout.strip()
    if output:
        return "PENDING" if "PD" in output else "RUNNING"
    else:
        return "COMPLETED"


def wait_for_job_completion(job_id: str):
    status = check_job_status(job_id)
    while status in ["PENDING", "RUNNING"]:
        print(f"Job {job_id} is {status}. Waiting...")
        time.sleep(60)  # Wait for 60 seconds before checking again
        status = check_job_status(job_id)
    print(f"Job {job_id} is {status}.")


def send_job(script_name: str, version: str, threshold: float) -> str:
    with open(f"{script_name}.sh", "r") as file:
        script = file.read()

    script = script.replace(f"${{{'VERSION'}}}", str(version))
    script = script.replace(f"${{{'THRESHOLD'}}}", str(threshold))
    tmp_script_name = f"{script_name}_tmp.sh"

    with open(tmp_script_name, "w") as file:
        file.write(script)

    job_id = submit_job(tmp_script_name)
    os.remove(tmp_script_name)
    return job_id


def run_sd():
    for version in VERSIONS:
        for threshold in THRESHOLDS:
            script_name = "template"
            job_id = send_job(script_name, version, threshold)
            wait_for_job_completion(job_id)


def run_clip(labels: List[str], clip_folder: str):
    dataframe = []
    percentages_strings = [
        f"% {label} in {suffix}"
        for label in labels
        for suffix in ["Original", "Output"]
    ]
    for version in VERSIONS:
        original_path = Path(f"images/{clip_folder}-sd_{version}/original")
        for threshold in THRESHOLDS:
            output_path = Path(f"images/{clip_folder}-sd_{version}/{threshold}")
            (
                percentages,
                original_clip_score,
                output_clip_score,
            ) = evaluate_images(original_path, output_path)

            dataframe.append(
                [
                    version,
                    threshold,
                    # original_clip_score,
                    # output_clip_score,
                ].extend(percentages)
            )


    df = pd.DataFrame(
        dataframe,
        columns=[
            "Stable Diffusion Version",
            "Threshold",
            # "Original CLIP Score",
            # "Output CLIP Score",
        ].extend(percentages_strings),
    )
    df.to_csv("clip_results.csv", index=False)


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--process",
        type=str,
        default="sd",
        choices=["sd", "clip"],
        help="The process to run.",
    )
    parser.add_argument(
        "--labels",
        type=list,
        default=["Pink Elephant", "Something Else"],
        help="The labels for zero-shot classification.",
    )
    parser.add_argument(
        "--clip_folder",
        type=str,
        default="without_pink_elephant",
        help="The folder name for CLIP evaluation.",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)

    args = parser.parse_args()
    if args.process == "sd":
        run_sd()
    else:
        run_clip(args.labels, args.clip_folder)
