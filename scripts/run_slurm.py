import subprocess
import os
from pathlib import Path


# SLURM job
slurm_job = """#!/bin/zsh
#SBATCH --job-name={name}
#SBATCH --output=logs/{name}-%j.out
#SBATCH --error=logs/{name}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,iaifi_gpu
"""


def create_job(cmd_base, name):
    slurm_scripts_path = Path("logs/scripts")
    os.makedirs(slurm_scripts_path, exist_ok=True)

    job = slurm_job.format(name=name) + cmd_base
    job_name = slurm_scripts_path / f"{name}.sh"
    with open(job_name, "w") as f:
        f.write(job)
    subprocess.run(["sbatch", job_name])