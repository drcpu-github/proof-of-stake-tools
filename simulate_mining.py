#!/usr/bin/python3

import datetime
import logging
import optparse
import os
import random
import sys
import time
import tqdm

import matplotlib.pyplot as plt

def select_logging_level(level):
    if level.lower() == "debug":
        return logging.DEBUG
    elif level.lower() == "info":
        return logging.INFO
    elif level.lower() == "warning":
        return logging.WARNING
    elif level.lower() == "error":
        return logging.ERROR
    elif level.lower() == "critical":
        return logging.CRITICAL
    else:
        print(f"Unknown logging level: {level}")
        sys.exit(1)

def configure_logger(log_filename, timestamp, log_level):
    logger = logging.getLogger()

    # Read filename details
    dirname = os.path.dirname(log_filename)
    filename, extension = os.path.splitext(os.path.basename(log_filename))
    # Add timestamp in log filename
    log_filename = os.path.join(dirname, f"{filename}.{timestamp}{extension}")
    # Setup file handler logging
    file_handler = logging.FileHandler(log_filename)

    # Set log level
    log_level = select_logging_level(log_level)
    logger.setLevel(log_level)

    # Add header formatting of the log message
    logging.Formatter.converter = time.gmtime
    formatter = logging.Formatter("[%(levelname)-8s] [%(asctime)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Add file handler
    logger.addHandler(file_handler)

    return logger

def build_stakers(logger, num_stakers, whales, whales_stake, total_staked, distribution, coin_ageing, timestamp):
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

    stakers = {i: int(s / sum(stakers) * (total_staked * (100 - whales_stake) / 100)) for i, s in enumerate(stakers)}
    num_whales = whales
    while whales > 0:
        stakers[num_stakers - whales] = int(whales_stake / 100 / num_whales * total_staked)
        whales -= 1

    logger.info(f"Stakers: {stakers}")

    plot_stakers(stakers, num_stakers, total_staked, distribution, coin_ageing, timestamp)

    return stakers

def plot_stakers(stakers, num_stakers, total_staked, distribution, coin_ageing, timestamp):
    fig, ax = plt.subplots(1, 1)
    ax.hist([stake / 1E6 for stake in stakers.values()], bins=32)
    ax.set_xlabel("Staked (M)")
    ax.set_ylabel("Number of stakers")
    plt.title(f"Stakers = {num_stakers}, total staked = {int(total_staked / 1E6)}M, distribution = {distribution}, ageing = {coin_ageing}")
    plt.savefig(f"plots/stakers_{timestamp}.png", bbox_inches="tight")

def simulate_epoch_vrf_stake(logger, epoch, stakers, coin_age, replication):
    logger.info(f"Simulating epoch {epoch + 1}")
    # eligibility: vrf < (2 ** 256) * own_power / global_power * rf

    # Calculate global power
    total_staked = sum(stakers.values())
    global_average_age = int(sum(coin_age.values()) / len(coin_age))
    global_power = min(total_staked * global_average_age, 18_446_744_073_709_551_615)

    proposals = []
    vrf = random.random()
    logger.info(f"VRF target: {vrf}")
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = min(stake * coin_age[staker], 147_573_952_589_676_416)
        eligibility = own_power / global_power * replication
        if eligibility > 1.0:
            logger.warning(f"Eligibility for staker {staker} exceeds 1.0: {eligibility}")
        if vrf < eligibility:
            block_hash = random.random()
            logger.info(f"Staker {staker} is eligible to propose a block: {vrf} < {eligibility}, {block_hash}")
            proposals.append((staker, block_hash))

    if len(proposals) == 0:
        logger.warning(f"No blocks proposed")
        return -1
    else:
        # Miner with the lowest VRF value is picked as the winner
        miner = min(proposals, key=lambda l: l[1])
        logger.info(f"Staker {miner[0]} is selected to propose a block: {miner[1]}")
        return miner[0]

def simulate_epoch_modulo_stake(logger, epoch, stakers, coin_age):
    logger.info(f"Simulating epoch {epoch + 1}")
    # eligibility: vrf % global_power == ordered_power_staker_x

    # Calculate global power
    global_power = sum(stake * coin_age[staker] for staker, stake in stakers.items())

    # equivalent to vrf (previous block hash + epoch) % global_power
    vrf = random.randint(0, global_power)
    logger.info(f"VRF target: {vrf}")

    cumulative_power, selected_staker = 0, -1
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = stake * coin_age[staker]
        if vrf >= cumulative_power and vrf < cumulative_power + own_power:
            logger.info(f"Staker {staker} is eligible to propose a block: {cumulative_power} <= {vrf} < {cumulative_power + own_power}")
            selected_staker = staker
        cumulative_power += own_power

    assert global_power == cumulative_power, f"{global_power} != {cumulative_power}"

    if selected_staker == -1:
        logger.warning("No blocks proposed")
    return selected_staker

def update_coin_age_reset(num_stakers, coin_age, miner):
    for staker in range(num_stakers):
        if staker != miner:
            coin_age[staker] += 1
        else:
            coin_age[staker] = 0
    return coin_age

def update_coin_age_halving(num_stakers, coin_age, miner):
    for staker in range(num_stakers):
        if staker != miner:
            coin_age[staker] += 1
        else:
            coin_age[staker] = int(coin_age[staker] / 2)
    return coin_age

def update_coin_age_capped(num_stakers, coin_age, miner):
    for staker in range(num_stakers):
        if staker != miner:
            coin_age[staker] = min(coin_age[staker] + 1, 1920)
        else:
            coin_age[staker] = 0
    return coin_age

def print_mining_stats(logger, stakers, mined_blocks, epochs):
    logger.info("Miner: blocks (percentage) -- stake (percentage)")
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=True):
        blocks = mined_blocks[staker] if staker in mined_blocks else 0
        block_percentage = blocks / epochs * 100
        stake_percentage = stake / sum(stakers.values()) * 100
        logger.info(f"{staker}: {blocks} ({block_percentage:.2f}%) -- {stake / 1E6:.2f}M ({stake_percentage:.2f}%)")

def plot_mining_rate(stakers, mined_blocks, epochs, num_stakers, total_staked, distribution, coin_ageing, timestamp):
    block_percentages, stake_percentages = [], []
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=True):
        blocks = mined_blocks[staker] if staker in mined_blocks else 0
        block_percentages.append(blocks / epochs * 100)
        stake_percentages.append(stake / sum(stakers.values()) * 100)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(stake_percentages, block_percentages)
    ax.set_xlabel("Staked (%)")
    ax.set_ylabel("Blocks mined (%)")
    plt.title(f"Stakers = {num_stakers}, total staked = {int(total_staked / 1E6)}M, distribution = {distribution}, ageing = {coin_ageing}")
    plt.savefig(f"plots/mining_{timestamp}.png", bbox_inches="tight")

def main():
    parser = optparse.OptionParser()
    parser.add_option("--stakers", type="int", default=100, dest="num_stakers")
    parser.add_option("--total-staked", type="int", default=100000000, dest="total_staked")
    parser.add_option("--distribution", type="string", default="random", dest="distribution")
    parser.add_option("--whales", type="int", default=0, dest="whales")
    parser.add_option("--whales-stake", type="int", default=0, dest="whales_stake")
    parser.add_option("--epochs", type="int", default=100000, dest="epochs")
    parser.add_option("--replication", type="int", default=16, dest="replication")
    parser.add_option("--coin-ageing", type="string", default="reset", dest="coin_ageing")
    parser.add_option("--mining-eligibility", type="string", default="vrf-stake", dest="mining_eligibility")
    parser.add_option("--logging", type="string", default="info", dest="logging")
    options, args = parser.parse_args()

    if not os.path.exists("plots"):
        os.mkdir("plots")

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    logger = configure_logger("mining", timestamp, options.logging)

    logger.info(f"Command line arguments: {options}")

    stakers = build_stakers(logger, options.num_stakers, options.whales, options.whales_stake, options.total_staked, options.distribution, options.coin_ageing, timestamp)
    # Everyone starts off with coin age equal to 1, otherwise no block will be mined in the first epoch
    coin_age = {staker: 1 for staker in range(options.num_stakers)}
    logger.info(f"Initial coin age: {coin_age}")

    mined_blocks = {}
    for epoch in tqdm.tqdm(range(options.epochs)):
        if options.mining_eligibility == "vrf-stake":
            miner = simulate_epoch_vrf_stake(logger, epoch, stakers, coin_age, options.replication)
        elif options.mining_eligibility == "modulo-stake":
            miner = simulate_epoch_modulo_stake(logger, epoch, stakers, coin_age)
        else:
            print("Unknown mining eligibility strategy")
            sys.exit(1)

        if options.coin_ageing == "disabled":
            pass
        elif options.coin_ageing == "reset":
            coin_age = update_coin_age_reset(options.num_stakers, coin_age, miner)
        elif options.coin_ageing == "halving":
            coin_age = update_coin_age_halving(options.num_stakers, coin_age, miner)
        elif options.coin_ageing == "capped":
            coin_age = update_coin_age_capped(options.num_stakers, coin_age, miner)
        else:
            print("Unknown coin ageing strategy")
            sys.exit(1)
        logger.info(f"Updated coin age: {coin_age}")

        if miner not in mined_blocks:
            mined_blocks[miner] = 0
        mined_blocks[miner] += 1

    print_mining_stats(logger, stakers, mined_blocks, options.epochs)

    plot_mining_rate(stakers, mined_blocks, options.epochs, options.num_stakers, options.total_staked, options.distribution, options.coin_ageing, timestamp)

if __name__ == "__main__":
    main()
