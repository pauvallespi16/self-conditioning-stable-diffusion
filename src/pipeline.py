import os
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from clip import evaluate_images

VERSIONS = ["1-1", "1-5", "2", "xl_1-0"]
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

    with open(f"{script_name}_tmp.sh", "w") as file:
        file.write(script)

    job_id = submit_job(f"{script_name}_tmp.sh")
    return job_id


def run_sd():
    for version in VERSIONS:
        for threshold in THRESHOLDS:
            job_id = send_job("pipeline", version, threshold)
            wait_for_job_completion(job_id)

    os.remove("pipeline_tmp.sh")


def run_clip():
    dataframe = []
    for version in VERSIONS:
        for threshold in THRESHOLDS:
            original_path = Path(
                f"images/without_pink_elephant-sd_{version}_{threshold}/original"
            )
            output_path = Path(
                f"images/without_pink_elephant-sd_{version}_{threshold}/output"
            )
            original_percentage, output_percentage = evaluate_images(
                original_path, output_path
            )
            dataframe.append(
                [
                    version.replace("-", "."),
                    threshold,
                    original_percentage,
                    output_percentage,
                ]
            )

    df = pd.DataFrame(
        dataframe,
        columns=[
            "Stable Diffusion Version",
            "Threshold",
            "% Pink Elephants in Original",
            "% Pink Elephants in Output",
        ],
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


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)

    args = parser.parse_args()
    if args.process == "sd":
        run_sd()
    else:
        run_clip()
