"""Replay the SHAFE experiments from the frozen per-scene records."""

from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parent
ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/hoseamuyi-bit/shafe-reproducibility/"
    "b2c7e9a3dabf3e764cee831ff14c77ff9e50f193/"
)
ARCHIVES = {
    "estimators": (
        "SHAFE_experiment_reproducibility.zip",
        "dd2cc42f992dd2e62cd28d9806c80e1bf228c2ea84ddb9b689aa351280f4ac55",
    ),
    "rankings": (
        "SHAFE模型排序确认_复现包.zip",
        "2006fa0d5cfc1702697e4450ccc848f6ab5bf64e577e34bc04b1bd00af70083a",
    ),
    "arkitscenes": (
        "SHAFE_ARKitScenes独立确认复算包.zip",
        "743bcc7030f8520e4a6bec402b5cedf6542e986ca262dd9d957157d3e412d51c",
    ),
}


def unpack_experiment(experiment, archive_directory, stage):
    filename, expected_digest = ARCHIVES[experiment]
    if archive_directory is None:
        archive_path = stage / filename
        with urlopen(ARCHIVE_BASE + quote(filename), timeout=60) as response:
            with archive_path.open("wb") as file:
                copyfileobj(response, file)
    else:
        archive_path = archive_directory / filename

    with archive_path.open("rb") as file:
        digest = file_digest(file, "sha256").hexdigest()
        if digest != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {archive_path}")
        root = stage / experiment
        with ZipFile(file) as archive:
            archive.extractall(root)
            members = {
                Path(name).name: name for name in archive.namelist()
                if name.endswith(".py")
            }
            for source in (PROJECT / experiment).glob("*.py"):
                if experiment == "estimators" and source.name == "dev_pipeline.py":
                    # The published module has a portable data path. Replay the frozen copy.
                    continue
                member = members[source.name]
                code = source.read_bytes()
                if code != archive.read(member):
                    raise ValueError(f"Source differs from the frozen copy: {source}")
                (root / member).write_bytes(code)
    return root


def run(root, report, *command):
    completed = subprocess.run(
        [sys.executable, "-E", *command], cwd=root, check=True, stdout=subprocess.PIPE,
    )
    report.write_bytes(completed.stdout)
    print(f"  {report.name}", flush=True)


def estimators(root, output):
    run(root, output / "estimators.json", "recompute.py", "results/formal_cases.csv",
        "--external", "results/external_cases.csv", "--transfer", "results/transfer_cases.csv")
    run(root, output / "frozen_replay.json", "experiment_audit/verify_frozen_replay.py", ".")
    run(root, output / "div2k_holdout.json", "experiment_audit/verify_holdout.py",
        "experiment_audit/holdout")
    run(root, output / "confirmation.json", "experiment_audit/verify_confirmation.py",
        "experiment_audit", "--pixel-replay")
    run(root, output / "registration_extensions.txt", "experiment_audit/verify_registration_extensions.py",
        "experiment_audit", "--pixel-replay")
    run(root, output / "pixel_metrics.txt", "experiment_audit/pixel_metric_audit.py", "verify")
    run(root, output / "rank_audit.txt", "experiment_audit/rank_audit.py",
        "results/formal_cases.csv", "results/transfer_cases.csv", str(output / "rank_audit.json"))


def rankings(root, output):
    run(root, output / "summary.json", "recompute_statistics.py", "--output", str(output / "analysis.json"))


def arkitscenes(root, output):
    run(root, output / "verification.txt", "shafe_public_optical/verify_confirmation_statistics.py",
        "shafe_public_optical/arkitscenes_temporal_confirmation", "--secondary", "--metrics", "--online")


REPLAYS = {"estimators": estimators, "rankings": rankings, "arkitscenes": arkitscenes}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=REPLAYS, help="Run one experiment group (default: all).")
    parser.add_argument("--archives", type=Path, help="Use local ZIPs from this directory instead of downloading them.")
    parser.add_argument("--output", type=Path, default=Path("reproduced"), help="Directory for replay reports.")
    args = parser.parse_args()
    output = args.output.resolve()
    experiments = [args.experiment] if args.experiment else REPLAYS
    with TemporaryDirectory(prefix="shafe-replay-") as temporary:
        for experiment in experiments:
            print(f"Recomputing {experiment}", flush=True)
            root = unpack_experiment(experiment, args.archives, Path(temporary))
            report_directory = output / experiment
            report_directory.mkdir(parents=True, exist_ok=True)
            REPLAYS[experiment](root, report_directory)
    print(f"Reports written to {output}", flush=True)


if __name__ == "__main__":
    main()
