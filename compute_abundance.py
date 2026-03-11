import argparse
import multiprocessing
import os
import numpy as np


def get_abundance(sample, number, fraction=0.1):
    print(sample)
    abundance = np.zeros(number, dtype=np.float64)
    for assign in sorted(os.listdir(sample)):
        if "assign" not in assign:
            continue
        assign_path = os.path.join(sample, assign)
        assigned = np.load(assign_path)
        if len(assigned) == 0:
            continue

        # Keep 10% of assignments to match current experiment settings.
        end = max(1, int(len(assigned) * fraction))
        for read in assigned[:end]:
            cluster_id = int(read[0])
            if 0 <= cluster_id < number:
                abundance[cluster_id] += 1

    total_assigned = np.sum(abundance)
    if total_assigned > 0:
        abundance = abundance / total_assigned
    else:
        print(f"No assignments found in {sample}; saving a zero vector.")

    np.save(os.path.join(sample, "abundance_10.npy"), abundance)
    print(sample, "saved")
    return abundance


def all_samples_parallel(samples_dir, number, processes=32):
    samples = sorted(os.listdir(samples_dir))
    sample_paths = [
        os.path.join(samples_dir, sample)
        for sample in samples
        if os.path.isdir(os.path.join(samples_dir, sample))
    ]
    if not sample_paths:
        print(f"No sample directories found in {samples_dir}")
        return

    worker_count = min(processes, os.cpu_count() or 1)
    with multiprocessing.Pool(processes=worker_count) as pool:
        pool.starmap(get_abundance, [(sample, number) for sample in sample_paths])


def all_numbers(numbers_dir, processes=32):
    numbers = sorted(os.listdir(numbers_dir))
    for number in numbers:
        if not number.isdigit():
            continue
        print(number)
        samples_dir = os.path.join(numbers_dir, number, "Fold_0", "all")
        if not os.path.isdir(samples_dir):
            print(f"Skipping missing directory: {samples_dir}")
            continue
        all_samples_parallel(samples_dir, int(number), processes=processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalized cluster abundances.")
    parser.add_argument(
        "numbers_dir",
        type=str,
        help="Directory that contains one folder per cluster count (e.g. 16, 32, ...).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=32,
        help="Number of worker processes to use.",
    )
    args = parser.parse_args()
    all_numbers(args.numbers_dir, processes=args.processes)
