#!/usr/bin/python3

import optparse
import os
import random
import sys
import tqdm

import matplotlib.pyplot as plt

def build_stakers(num_stakers, total_staked, distribution):
    if distribution == "random":
        stakers = [random.random() for s in range(num_stakers)]
    elif distribution == "uniform":
        stakers = [total_staked / num_stakers for s in range(num_stakers)]
    elif distribution == "beta":
        stakers = [random.betavariate(0.5, 0.2) for s in range(num_stakers)]
    elif distribution == "exponential":
        stakers = [random.expovariate(2) for s in range(num_stakers)]
    elif distribution == "gamma":
        stakers = [random.gammavariate(0.5, 0.2) for s in range(num_stakers)]
    else:
        print("Unknown staking distribution")
        sys.exit(1)

    stakers = {i: int(s / sum(stakers) * total_staked) for i, s in enumerate(stakers)}

    fig, ax = plt.subplots(1, 1)
    ax.hist([stake / 1E6 for stake in stakers.values()])
    ax.set_xlabel("Staked (M)")
    ax.set_ylabel("Number of stakers")
    plt.title(f"Stakers = {num_stakers}, total staked = {int(total_staked / 1E6)}M, distribution = {distribution}")
    plt.savefig(f"plots/stakers_stakers={num_stakers}_staked={int(total_staked / 1E6)}M_distribution={distribution}.png", bbox_inches="tight")

    return stakers

def simulate_epoch(stakers, coin_age, replication):
    # eligibility: vrf < (2 ** 256) * own_power / global_power * rf

    # Calculate global power
    total_staked = sum(stakers.values())
    global_average_age = int(sum(coin_age.values()) / len(coin_age))
    global_power = min(total_staked * global_average_age, 184_467_440_737_095_551_615)

    eligibility = []
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = min(stake * coin_age[staker], 1_475_739_525_896_764_412)

        vrf = random.random()
        if vrf < own_power / global_power * replication:
            eligibility.append((staker, vrf))

    if len(eligibility) == 0:
        return -1
    else:
        # Miner with the lowest VRF value is picked as the winner
        miner = min(eligibility, key=lambda l: l[1])
        return miner[0]

def update_coin_age(num_stakers, coin_age, miner):
    for staker in range(num_stakers):
        if staker != miner:
            coin_age[staker] += 1
        else:
            coin_age[staker] = 0
    return coin_age

def main():
    parser = optparse.OptionParser()
    parser.add_option("--stakers", type="int", default=100, dest="num_stakers")
    parser.add_option("--total-staked", type="int", default=100000000, dest="total_staked")
    parser.add_option("--distribution", type="string", default="random", dest="distribution")
    parser.add_option("--epochs", type="int", default=100000, dest="epochs")
    parser.add_option("--replication", type="int", default=16, dest="replication")
    parser.add_option("--verbose", action="store_true", dest="verbose")
    options, args = parser.parse_args()

    if not os.path.exists("plots"):
        os.mkdir("plots")

    stakers = build_stakers(options.num_stakers, options.total_staked, options.distribution)
    # Everyone starts off with coin age equal to 1, otherwise no block will be mined in the first epoch
    coin_age = {staker: 1 for staker in range(options.num_stakers)}

    previous_miner = -1
    mined_blocks = {}
    for epoch in tqdm.tqdm(range(options.epochs)):
        miner = simulate_epoch(stakers, coin_age, options.replication)
        if miner != -1:
            assert miner != previous_miner, "No miner should be able to mine two subsequent blocks"
        previous_miner = miner

        coin_age = update_coin_age(options.num_stakers, coin_age, miner)

        if miner not in mined_blocks:
            mined_blocks[miner] = 0
        mined_blocks[miner] += 1

    block_percentages, stake_percentages = [], []
    if options.verbose:
        print("Miner: blocks (percentage) -- stake (percentage)")
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=True):
        blocks = mined_blocks[staker] if staker in mined_blocks else 0
        block_percentage = blocks / options.epochs * 100
        stake_percentage = stake / sum(stakers.values()) * 100

        block_percentages.append(block_percentage)
        stake_percentages.append(stake_percentage)

        if options.verbose:
            print(f"{staker}: {blocks} ({block_percentage:.2f}%) -- {stake / 1E6:.2f}M ({stake_percentage:.2f}%)")

    fig, ax = plt.subplots(1, 1)
    ax.scatter(stake_percentages, block_percentages)
    ax.set_xlabel("Staked (%)")
    ax.set_ylabel("Blocks mined (%)")
    plt.title(f"Stakers = {options.num_stakers}, total staked = {int(options.total_staked / 1E6)}M, distribution = {options.distribution}")
    plt.savefig(f"plots/mining_stakers={options.num_stakers}_staked={int(options.total_staked / 1E6)}M_distribution={options.distribution}.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
