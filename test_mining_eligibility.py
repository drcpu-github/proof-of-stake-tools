#!/usr/bin/python3

import optparse
import random
import sys
import tqdm

import matplotlib.pyplot as plt

def build_stakers(num_stakers, distribution):
    if distribution == "gauss":
        stakers = [random.gauss(0.5, 0.2) for n in range(num_stakers)]
    elif distribution == "uniform":
        stakers = [random.random() for s in range(num_stakers)]
    elif distribution == "beta":
        stakers = [random.betavariate(0.5, 0.2) for s in range(num_stakers)]
    elif distribution == "exponential":
        stakers = [random.expovariate(2) for s in range(num_stakers)]
    elif distribution == "gamma":
        stakers = [random.gammavariate(0.5, 0.2) for s in range(num_stakers)]
    else:
        print("Unknown staking distribution")
        sys.exit(1)

    stakers = [s / sum(stakers) for s in stakers]

    fig, ax = plt.subplots(1, 1)
    ax.hist(stakers)
    plt.savefig(f"stakers_{num_stakers}_{distribution}.png", bbox_inches="tight")

    return stakers

def simulate_epoch(stakers, replication):
    # eligibility: vrf < (2 ** 256) * own_power / global_power * rf
    vrf = random.random()
    eligible = [1 for staker in stakers if vrf < staker * replication]
    return sum(eligible)

def main():
    parser = optparse.OptionParser()
    parser.add_option("--stakers", type="int", default=1000, dest="stakers")
    parser.add_option("--distribution", type="string", default="gauss", dest="distribution")
    parser.add_option("--epochs", type="int", default=100000, dest="epochs")
    parser.add_option("--replication", type="int", default=8, dest="replication")
    options, args = parser.parse_args()

    stakers = build_stakers(options.stakers, options.distribution)

    candidates_histogram = {}
    for epoch in tqdm.tqdm(range(options.epochs)):
        block_candidates = simulate_epoch(stakers, options.replication)
        if block_candidates not in candidates_histogram:
            candidates_histogram[block_candidates] = 0
        candidates_histogram[block_candidates] += 1

    print("#Candidates: times (percentage)")
    for candidates, times in sorted(candidates_histogram.items()):
        print(f"{candidates}: {times} ({times / options.epochs * 100:.2f}%)")

if __name__ == "__main__":
    main()
