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
from matplotlib.ticker import PercentFormatter

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
    log_filename = os.path.join(dirname, f"logs/{filename}.{timestamp}{extension}")
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

def build_stakers(logger, options, timestamp):
    num_stakers = options.num_stakers
    whales = options.whales
    whales_stake = options.whales_stake
    total_staked = options.total_staked
    distribution = options.distribution
    coin_ageing = options.coin_ageing
    mining_eligibility = options.mining_eligibility

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

    plot_stakers(stakers, num_stakers, total_staked, distribution, coin_ageing, mining_eligibility, timestamp)

    return stakers

def plot_stakers(stakers, num_stakers, total_staked, distribution, coin_ageing, mining_eligibility, timestamp):
    fig, ax = plt.subplots(1, 1)
    ax.hist([stake / 1E6 for stake in stakers.values()], bins=32)
    ax.set_xlabel("Staked (M)")
    ax.set_ylabel("Number of stakers")
    plt.title(f"Stakers = {num_stakers}, total staked = {int(total_staked / 1E6)}M, distribution = {distribution}, ageing = {coin_ageing}, mining = {mining_eligibility}")
    plt.savefig(f"plots/stakers_{timestamp}.png", bbox_inches="tight")

def simulate_epoch_vrf_stake(logger, epoch, stakers, coin_age, replication):
    logger.info(f"Simulating epoch {epoch + 1}")
    # eligibility: vrf < (2 ** 256) * own_power / global_power * rf

    # Calculate global power
    total_staked = sum(stakers.values())
    global_average_age = int(sum(coin_age.values()) / len(coin_age))
    global_power = min(total_staked * global_average_age, 18_446_744_073_709_551_615)

    proposals = []
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = min(stake * coin_age[staker], 147_573_952_589_676_416)
        eligibility = own_power / global_power * replication
        if eligibility > 1.0:
            logger.warning(f"Eligibility for staker {staker} exceeds 1.0: {eligibility}")
        vrf = random.random()
        if vrf < eligibility:
            logger.info(f"Staker {staker} is eligible to propose a block: {vrf} < {eligibility}")
            proposals.append((staker, vrf))

    if len(proposals) == 0:
        logger.warning(f"No blocks proposed")
        return -1, []
    else:
        # Miner with the lowest VRF value is picked as the winner
        miner = min(proposals, key=lambda l: l[1])
        logger.info(f"Staker {miner[0]} is selected to propose a block: {miner[1]}")
        return miner[0], proposals

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
            assert selected_staker == -1
            selected_staker = staker
        cumulative_power += own_power

    assert global_power == cumulative_power, f"{global_power} != {cumulative_power}"

    if selected_staker == -1:
        logger.warning("No blocks proposed")
    return selected_staker, 1 if selected_staker > -1 else 0

def simulate_epoch_modulo_slot(logger, epoch, stakers, coin_age, replication):
    logger.info(f"Simulating epoch {epoch + 1}")
    # eligibility: vrf % (global_power / replication) == ordered_power_staker_x

    # Calculate global power
    global_power = sum(stake * coin_age[staker] for staker, stake in stakers.items())

    # equivalent to (previous block hash + epoch) % (global_power / replication)
    global_power_slot_size = int(global_power / replication)
    vrf = random.randint(0, global_power_slot_size - 1)
    logger.info(f"VRF target: {vrf}")

    proposals = []
    cumulative_power, staker = 0, 0
    while cumulative_power < global_power:
        staker_power = stakers[staker] * coin_age[staker]

        start_slot = int(cumulative_power / global_power_slot_size)
        end_slot = int((cumulative_power + staker_power) / global_power_slot_size)
        for s in range(start_slot, end_slot + 1):
            power_lower_bound = max(s * global_power_slot_size, cumulative_power) % global_power_slot_size
            power_upper_bound = min(cumulative_power + staker_power, (s + 1) * global_power_slot_size - 1) % global_power_slot_size
            if vrf >= power_lower_bound and vrf < power_upper_bound:
                logger.info(f"Staker {staker} is eligible to propose a block: {power_lower_bound} <= {vrf} < {power_upper_bound} (slot {s})")
                proposals.append((staker, random.random()))

        cumulative_power += staker_power
        staker += 1

    assert len(proposals) == replication

    # Miner with the lowest block hash value is picked as the winner
    miner = min(proposals, key=lambda l: l[1])
    logger.info(f"Staker {miner[0]} is selected to propose a block: {miner[1]}")
    return miner[0], len(proposals)

def update_coin_age_reset(num_stakers, coin_age, miner):
    for staker in range(num_stakers):
        if staker != miner:
            coin_age[staker] = min(coin_age[staker] + 1, 53760)
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

def plot_mining_rate(stakers, mined_blocks, epochs, plot_title, timestamp):
    block_percentages, stake_percentages = [], []
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=True):
        blocks = mined_blocks[staker] if staker in mined_blocks else 0
        block_percentages.append(blocks / epochs * 100)
        stake_percentages.append(stake / sum(stakers.values()) * 100)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(stake_percentages, block_percentages)
    ax.set_xlabel("Staked (%)")
    ax.set_ylabel("Blocks mined (%)")
    plt.title(plot_title)
    plt.savefig(f"plots/mining_{timestamp}.png", bbox_inches="tight")

def plot_num_blocks_proposed(num_blocks_proposed, plot_title, timestamp):
    x_values, y_values = [], []
    for x, y in num_blocks_proposed.items():
        x_values.append(x)
        y_values.append(y / sum(num_blocks_proposed.values()))

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(max(10, 0.5 * len(x_values)), 8)
    ax.bar(x_values, y_values)
    ax.xaxis.set_ticks(range(min(x_values), max(x_values) + 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(plot_title)
    plt.savefig(f"plots/blocks_{timestamp}.png", bbox_inches="tight")

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
    parser.add_option("--block-reward", type="int", default=250, dest="block_reward")
    parser.add_option("--logging", type="string", default="info", dest="logging")
    options, args = parser.parse_args()

    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists("logs"):
        os.mkdir("logs")

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    logger = configure_logger("mining", timestamp, options.logging)

    logger.info(f"Command line arguments: {options}")

    stakers = build_stakers(logger, options, timestamp)
    # Everyone starts off with coin age equal to 1, otherwise no block will be mined in the first epoch
    coin_age = {staker: 1 for staker in range(options.num_stakers)}
    logger.info(f"Initial coin age: {coin_age}")

    mined_blocks = {}
    no_blocks_proposed = 0
    num_blocks_proposed = {}
    staker_block_proposed = {}
    for epoch in tqdm.tqdm(range(options.epochs)):
        if options.mining_eligibility == "vrf-stake":
            miner, proposals = simulate_epoch_vrf_stake(logger, epoch, stakers, coin_age, options.replication)
            for proposer, vrf in proposals:
                if proposer not in staker_block_proposed:
                    staker_block_proposed[proposer] = 0
                staker_block_proposed[proposer] += 1
            num_proposals = len(proposals)
        elif options.mining_eligibility == "modulo-stake":
            miner, num_proposals = simulate_epoch_modulo_stake(logger, epoch, stakers, coin_age)
        elif options.mining_eligibility == "modulo-slot":
            miner, num_proposals = simulate_epoch_modulo_slot(logger, epoch, stakers, coin_age, options.replication)
        else:
            print("Unknown mining eligibility strategy")
            sys.exit(1)

        if miner != -1:
            stakers[miner] += options.block_reward

        if miner == -1:
            no_blocks_proposed += 1

        if miner not in mined_blocks:
            mined_blocks[miner] = 0
        mined_blocks[miner] += 1

        if num_proposals not in num_blocks_proposed:
            num_blocks_proposed[num_proposals] = 0
        num_blocks_proposed[num_proposals] += 1

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

    print_mining_stats(logger, stakers, mined_blocks, options.epochs)
    if no_blocks_proposed > options.epochs * 0.01:
        no_proposal_percentage = no_blocks_proposed / options.epochs * 100
        print(f"WARNING: no blocks proposed for {no_blocks_proposed} epochs ({no_proposal_percentage:.2f}% of total)")

    plot_title = f"Stakers = {options.num_stakers}, total staked = {int(options.total_staked / 1E6)}M, distribution = {options.distribution}, ageing = {options.coin_ageing}, mining = {options.mining_eligibility}"
    plot_mining_rate(stakers, mined_blocks, options.epochs, plot_title, timestamp)
    plot_num_blocks_proposed(num_blocks_proposed, plot_title, timestamp)

    if options.mining_eligibility == "vrf-stake":
        for staker, num_proposed in staker_block_proposed.items():
            logger.info(f"Staker {staker} proposed {num_proposed} blocks")

if __name__ == "__main__":
    main()
