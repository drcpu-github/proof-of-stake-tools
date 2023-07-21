#!/usr/bin/python3

import datetime
import logging
import numpy
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
    num_stakers = options.num_commons + options.num_whales

    distribution = options.distribution
    
    if distribution == "random":
        commons = [random.random() for s in range(options.num_commons)]
    elif distribution == "uniform":
        commons = [options.commons_staked / num_stakers for s in range(options.num_commons)]
    elif distribution == "beta":
        commons = [random.betavariate(0.5, 0.2) for s in range(options.num_commons)]
    elif distribution == "exponential":
        commons = [random.expovariate(2) for s in range(options.num_commons)]
    elif distribution == "gamma":
        commons = [random.gammavariate(0.5, 0.2) for s in range(options.num_commons)]
    else:
        print("Unknown staking distribution")
        sys.exit(1)

    stakers = {
        i: int(commons[i] / sum(commons) * options.commons_staked) 
            if i < options.num_commons 
            else options.commons_staked * options.whales_stake_percentage / 100 / options.num_whales 
        for i in range(num_stakers)
    }

    logger.info(f"Stakers: {stakers}")
    plot_stakers(stakers, len(stakers), sum(stakers.values()), options, timestamp)
    
    return stakers

def simulate_epoch_vrf_stake(logger, epoch, stakers, coin_age, replication, replication_selector):
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
            # Miner with the lowest VRF value will be picked as the winner
            if replication_selector == "lowest-hash":
                proposals.append((staker, vrf))
            # Miner with the highest power will be picked as the winner
            elif replication_selector == "highest-power":
                proposals.append((staker, own_power))
            else:
                print("Unknown replication selector")
                sys.exit(1)

    if len(proposals) == 0:
        logger.warning(f"No blocks proposed")
        return -1, []
    else:
        if replication_selector == "lowest-hash":
            miner = min(proposals, key=lambda l: l[1])
        elif replication_selector == "highest-power":
            miner = max(proposals, key=lambda l: l[1])
        else:
            print("Unknown replication selector")
            sys.exit(1)
        logger.info(f"Staker {miner[0]} is selected to propose a block: {miner[1]}")
        return miner[0], proposals

def simulate_epoch_vrf_stake_adaptative(logger, epoch, stakers, coin_age, replication, replication_selector):
    logger.info(f"Simulating epoch {epoch + 1}")
    # eligibility: vrf < (2 ** 256) * own_power / global_power * rf

    # Calculate global power
             
    powers = [ min(stake * coin_age[staker], 147_573_952_589_676_416) for staker, stake in stakers.items() ]
    global_power = max(powers)   
    num_stakers = len(stakers)
    threshold_power = numpy.quantile(powers, 1 - replication / num_stakers) if replication < num_stakers else 0
    
    logger.info(f"Threshold vs Global power: {threshold_power} vs {global_power} ({threshold_power / global_power * 100}%)")

    proposals = []
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = min(stake * coin_age[staker], 147_573_952_589_676_416)  
        eligibility = own_power / global_power * replication if own_power >= threshold_power else 0
        if eligibility > 1.0:
            logger.warning(f"Eligibility for staker {staker} exceeds 1.0: {eligibility}")
        vrf = random.random()
        if vrf < eligibility:
            logger.info(f"Staker {staker} is eligible to propose a block: {vrf} < {eligibility}")
            # Miner with the lowest VRF value will be picked as the winner
            if replication_selector == "lowest-hash":
                proposals.append((staker, vrf))
            # Miner with the highest power will be picked as the winner
            elif replication_selector == "highest-power":
                proposals.append((staker, own_power, vrf))
            else:
                print("Unknown replication selector")
                sys.exit(1)

    if len(proposals) == 0:
        logger.warning(f"No blocks proposed")
        return -1, []
    else:
        if replication_selector == "lowest-hash":
            miner = min(proposals, key=lambda l: l[1])
        elif replication_selector == "highest-power":
            miner = max(proposals, key=lambda l: l[1])
            # select candidates from those having maximum power (there could be many),
            # ultimately selecting the one having the highest vrf value
            candidates = map(lambda p: (p[0], p[2] if p[1] == miner[1] else 0), proposals)
            proposals = []
            for candidate in candidates:
                proposals.append(candidate)
            miner = max(proposals, key=lambda l: l[1])

        else:
            print("Unknown replication selector")
            sys.exit(1)
        logger.info(f"Staker {miner[0]} is selected to propose a block: {miner[1]}")
        return miner[0], proposals

def simulate_epoch_modulo_stake(logger, epoch, stakers, coin_age):
    logger.info(f"Simulating epoch {epoch + 1}")
    # eligibility: random_value % global_power == ordered_power_staker_x

    # Calculate global power
    global_power = sum(stake * coin_age[staker] for staker, stake in stakers.items())

    # equivalent to (previous block hash + epoch) % global_power
    random_slot = random.randint(0, global_power)
    logger.info(f"Random slot: {random_slot}")

    cumulative_power, selected_staker = 0, -1
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = stake * coin_age[staker]
        if random_slot >= cumulative_power and random_slot < cumulative_power + own_power:
            logger.info(f"Staker {staker} is eligible to propose a block: {cumulative_power} <= {random_slot} < {cumulative_power + own_power}")
            assert selected_staker == -1
            selected_staker = staker
        cumulative_power += own_power

    assert global_power == cumulative_power, f"{global_power} != {cumulative_power}"

    if selected_staker == -1:
        logger.warning("No blocks proposed")
    return selected_staker, 1 if selected_staker > -1 else 0

def simulate_epoch_modulo_slot(logger, epoch, stakers, coin_age, replication, replication_selector):
    logger.info(f"Simulating epoch {epoch + 1}")
    # eligibility: random_value % (global_power / replication) == ordered_power_staker_x

    # Calculate global power
    global_power = sum(stake * coin_age[staker] for staker, stake in stakers.items())

    # equivalent to (previous block hash + epoch) % (global_power / replication)
    global_power_slot_size = int(global_power / replication)
    random_slot = random.randint(0, global_power_slot_size - 1)
    logger.info(f"Random slot: {random_slot}")

    proposals = []
    cumulative_power, staker = 0, 0
    while cumulative_power < global_power:
        staker_power = stakers[staker] * coin_age[staker]

        start_slot = int(cumulative_power / global_power_slot_size)
        end_slot = int((cumulative_power + staker_power) / global_power_slot_size)
        for s in range(start_slot, end_slot + 1):
            power_lower_bound = max(s * global_power_slot_size, cumulative_power) % global_power_slot_size
            power_upper_bound = min(cumulative_power + staker_power, (s + 1) * global_power_slot_size - 1) % global_power_slot_size
            if random_slot >= power_lower_bound and random_slot < power_upper_bound:
                logger.info(f"Staker {staker} is eligible to propose a block: {power_lower_bound} <= {random_slot} < {power_upper_bound} (slot {s})")
                # Miner with the lowest block hash value will be picked as the winner
                if replication_selector == "lowest-hash":
                    proposals.append((staker, random.random()))
                # Miner with the highest power will be picked as the winner
                elif replication_selector == "highest-power":
                    proposals.append((staker, staker_power))
                else:
                    print("Unknown replication selector")
                    sys.exit(1)

        cumulative_power += staker_power
        staker += 1

    assert len(proposals) == replication

    if replication_selector == "lowest-hash":
        miner = min(proposals, key=lambda l: l[1])
    elif replication_selector == "highest-power":
        miner = max(proposals, key=lambda l: l[1])
    else:
        print("Unknown replication selector")
        sys.exit(1)
    logger.info(f"Staker {miner[0]} is selected to propose a block: {miner[1]}")
    return miner[0], len(proposals)

def update_coin_age_reset(num_stakers, coin_age, miner, options):
    for staker in range(num_stakers):
        if staker != miner:
            coin_age[staker] = min(coin_age[staker] + 1, 53760)
        else:
            coin_age[staker] = 0
    return coin_age

def update_coin_age_halving(num_stakers, coin_age, miner, options):
    for staker in range(num_stakers):
        if staker != miner:
            coin_age[staker] += 1
        else:
            coin_age[staker] = int(coin_age[staker] / 2)
    return coin_age

def update_coin_age_capped(num_stakers, coin_age, miner, options):
    for staker in range(num_stakers):
        if staker != miner:
            # max age capped to number of stakers
            coin_age[staker] = min(coin_age[staker] + 1, num_stakers * pow(options.replication, options.replication_power))
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

def plot_stakers(stakers, num_stakers, total_staked, options, timestamp):
    fig, ax = plt.subplots(1, 1)
    ax.hist([stake / 1E6 for stake in stakers.values()], bins=64)
    ax.set_xlabel("Staked (M)")
    ax.set_ylabel("Number of stakers")
    whales_staked = round(options.commons_staked * options.whales_stake_percentage / 100)
    plt.title(f"total stake = {round(total_staked / 1E6)}M, commons = {options.distribution}({num_stakers - options.num_whales}, {round((total_staked - whales_staked) / 1E6)}M), whales = uniform({options.num_whales}, {int(whales_staked/1E6)}M)")
    plt.savefig(f"plots/stakers_{plots_file_prefix(options)}_{timestamp}.png", bbox_inches="tight")

def plot_mining_rate(stakers, mined_blocks, options, plot_title, timestamp):
    block_percentages, stake_percentages = [], []
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=True):
        blocks = mined_blocks[staker] if staker in mined_blocks else 0
        block_percentages.append(blocks / options.epochs * 100)
        stake_percentages.append(stake / sum(stakers.values()) * 100)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(stake_percentages, block_percentages)
    ax.set_xlabel("Staked (%)")
    ax.set_ylabel("Blocks mined (%)")
    plt.title(plot_title)
    plt.savefig(f"plots/mining_{plots_file_prefix(options)}_{timestamp}.png", bbox_inches="tight")

def plot_num_blocks_proposed(num_blocks_proposed, options, plot_title, timestamp):
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
    plt.savefig(f"plots/blocks_{plots_file_prefix(options)}_{timestamp}.png", bbox_inches="tight")

def plots_file_prefix(options):
    return f"{options.mining_eligibility}_{options.coin_ageing}_{options.replication_selector}_{options.replication}_{options.replication_power}_{options.distribution}_{str(options.num_commons).rjust(4, '0')}_{str(int(options.whales_stake_percentage)).rjust(2, '0')}_{str(options.num_whales).rjust(3, '0')}"

def main():
    parser = optparse.OptionParser()
    parser.add_option("--stakers", type="int", default=100, dest="num_commons")
    parser.add_option("--initial-stake", type="int", default=1_000_000_000, dest="commons_staked")
    parser.add_option("--whales", type="int", default=0, dest="num_whales")
    parser.add_option("--whales-stake-increment", type="float", default=0, dest="whales_stake_percentage")
    parser.add_option("--distribution", type="string", default="random", dest="distribution")
    parser.add_option("--epochs", type="int", default=100000, dest="epochs")
    parser.add_option("--replication", type="int", default=16, dest="replication")
    parser.add_option("--replication-power", type="float", default=1, dest="replication_power")
    parser.add_option("--coin-ageing", type="string", default="reset", dest="coin_ageing")
    parser.add_option("--mining-eligibility", type="string", default="vrf-stake-adaptative", dest="mining_eligibility")
    parser.add_option("--replication-selector", type="string", default="lowest-hash", dest="replication_selector")
    parser.add_option("--block-reward", type="int", default=250, dest="block_reward")
    parser.add_option("--logging", type="string", default="info", dest="logging")
    parser.add_option("--max-staking-txs-per-block", type="int", default="32", dest="max_staking_txs_per_block")
    options, args = parser.parse_args()

    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists("logs"):
        os.mkdir("logs")

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    short_timestamp = datetime.datetime.now().strftime("%H%M%S")

    logger = configure_logger("mining", timestamp, options.logging)

    logger.info(f"Command line arguments: {options}")

    stakers = build_stakers(logger, options, timestamp)
    num_stakers = len(stakers)
    # all stakers but whales start off with coin age equal to maximum age (total number of stakers)
    coin_age = {
        staker: 
            len(stakers) * options.replication if staker < options.num_commons 
            else int((staker - options.num_commons) / options.max_staking_txs_per_block)
        for staker in range(num_stakers)}
    logger.info(f"Initial coin age: {coin_age}")

    mined_blocks = {}
    no_blocks_proposed = 0
    num_blocks_proposed = {}
    staker_block_proposed = {}
    first_block_epoch = {staker: 0 for staker in range(num_stakers)}
    for epoch in tqdm.tqdm(range(options.epochs)):
        if options.mining_eligibility.startswith("vrf-stake"):
            if options.mining_eligibility == "vrf-stake-adaptative":
                miner, proposals = simulate_epoch_vrf_stake_adaptative(logger, epoch, stakers, coin_age, options.replication, options.replication_selector)
            else:
                miner, proposals = simulate_epoch_vrf_stake(logger, epoch, stakers, coin_age, options.replication, options.replication_selector)
            for proposer, vrf in proposals:
                if proposer not in staker_block_proposed:
                    staker_block_proposed[proposer] = 0
                staker_block_proposed[proposer] += 1
            num_proposals = len(proposals)
        elif options.mining_eligibility == "modulo-stake":
            miner, num_proposals = simulate_epoch_modulo_stake(logger, epoch, stakers, coin_age)
        elif options.mining_eligibility == "modulo-slot":
            miner, num_proposals = simulate_epoch_modulo_slot(logger, epoch, stakers, coin_age, options.replication, options.replication_selector)
        else:
            print("Unknown mining eligibility strategy")
            sys.exit(1)

        if miner != -1:
            stakers[miner] += options.block_reward

        if miner == -1:
            no_blocks_proposed += 1
        else:
            if miner not in mined_blocks:
                mined_blocks[miner] = 0
            mined_blocks[miner] += 1

            if first_block_epoch[miner] == 0:
                first_block_epoch[miner] = epoch

        if num_proposals not in num_blocks_proposed:
            num_blocks_proposed[num_proposals] = 0
        num_blocks_proposed[num_proposals] += 1

        if options.coin_ageing == "disabled":
            pass
        elif options.coin_ageing == "reset":
            coin_age = update_coin_age_reset(num_stakers, coin_age, miner, options)
        elif options.coin_ageing == "halving":
            coin_age = update_coin_age_halving(num_stakers, coin_age, miner, options)
        elif options.coin_ageing == "capped":
            coin_age = update_coin_age_capped(num_stakers, coin_age, miner, options)
        else:
            print("Unknown coin ageing strategy")
            sys.exit(1)
        logger.info(f"Updated coin age: {coin_age}")

    print_mining_stats(logger, stakers, mined_blocks, options.epochs)

    whales_staked = options.commons_staked * options.whales_stake_percentage / 100
    total_staked = options.commons_staked + whales_staked

    if options.num_whales > 0:
        plot_title = f"commons = {options.num_commons} ({int(options.commons_staked / 1E6)}M), whales = {options.num_whales} ({int(whales_staked / 1E6)}M), mstb = {options.max_staking_txs_per_block}, rf = {options.replication}, rf-power = {options.replication_power}"
    else:
        plot_title = f"commons = {options.num_commons} ({int(options.commons_staked / 1E6)}M), mstb = {options.max_staking_txs_per_block}, rf = {options.replication}, rf-power = {options.replication_power}"
    plot_mining_rate(stakers, mined_blocks, options, plot_title, short_timestamp)
    plot_num_blocks_proposed(num_blocks_proposed, options, plot_title, short_timestamp)

    if options.mining_eligibility.startswith("vrf-stake"):
        for staker, num_proposed in staker_block_proposed.items():
            logger.info(f"Staker {staker} proposed {num_proposed} blocks")

    print(f"\nSimulation parameters:")
    print(f"> Number of epochs:      ", options.epochs)
    print(f"> Mining eligiblity:     ", options.mining_eligibility)
    print(f"> Coin ageing:           ", options.coin_ageing)
    print(f"> Replication selector:  ", options.replication_selector)
    print(f"> Replication factor:    ", options.replication)
    if options.replication > 1 and options.coin_ageing == "capped":
        print(f"> Repl. ageing power:    ", options.replication_power)
    print(f"> Commons distribution:  ", options.distribution)
    print( "> Commons initial stake: ", f"{int(options.commons_staked/1E6):,} MWIT (÷ {options.num_commons:,} nodes)")
    if (options.num_whales > 0):
        print( "> Whales incoming stake: ", f"+{whales_staked / options.commons_staked * 100:.2f}% => {int(total_staked/1E6):,} MWIT")
        percentage_str = f"{whales_staked / total_staked * 100:.2f}"
        print( "> Whales unitary stake:  ", f"{int(whales_staked/options.num_whales):,} WIT (x {options.num_whales:,} nodes)")
        
        
    print(f"\nSimulation results:") 

    void_blocks_percentage = no_blocks_proposed / options.epochs * 100
    percentage_str = f"{void_blocks_percentage:.2f}"
    print(f"> Void blocks percentage: {percentage_str.rjust(6)} % ({no_blocks_proposed:,} blocks)")

    underline_stake = 0
    num_underliners = 0
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=False):
        if staker not in mined_blocks or mined_blocks[staker] / options.epochs <= stake / total_staked / 10:
            num_underliners += 1
            if stake > underline_stake:
                underline_stake = stake
    
    percentage_str = f"{underline_stake / total_staked * 100:.2f}"
    print(f"> Underliners threshold:  {percentage_str.rjust(6)} % ({int(underline_stake):,} WIT)")
    percentage_str = f"{num_underliners / num_stakers * 100:.2f}"
    print(f"> Underliners percentile: {percentage_str.rjust(6)} % ({num_underliners:,} nodes)")

    if options.num_whales > 0:
        whales_first_epoch = options.epochs
        for staker, first_epoch in first_block_epoch.items():
            if staker >= options.num_commons:
                if first_epoch != 0 and first_epoch < whales_first_epoch:
                    whales_first_epoch = first_epoch

        whales_mined_blocks = 0
        for staker, num_mined in mined_blocks.items():
            if staker >= options.num_commons:
                whales_mined_blocks += num_mined

        percentage_str = f"{whales_staked / total_staked * 100:.2f}"
        print(f"> Whales relative stake:  {percentage_str.rjust(6)} % ({options.num_whales/num_stakers*100:.1f}% of nodes)")
        percentage_str = f"{whales_mined_blocks / options.epochs * 100:.2f}"
        print(f"> Whales elegibility:     {percentage_str.rjust(6)} % (after {whales_first_epoch:,} epochs)")                    
        save_simulation_results(short_timestamp, options, num_stakers, total_staked, void_blocks_percentage, underline_stake, num_underliners, whales_mined_blocks, whales_first_epoch)
    else:
        save_simulation_results(short_timestamp, options, num_stakers, total_staked, void_blocks_percentage, underline_stake, num_underliners, 0, options.epochs)
        
def save_simulation_results(index, options, num_stakers, total_staked, underline_stake, void_blocks_percentage, num_underliners, whales_mined_blocks, whales_first_epoch):
    csv_filename = f"results/{options.mining_eligibility}_{options.coin_ageing}_{options.replication_selector}_{options.distribution}.csv"
    modus_ponens = f"\"{options.mining_eligibility}\";\"{options.coin_ageing}\";\"{options.replication_selector}\";\"{options.distribution}\";"
    input_params = f"\"{index}\";\"{options.replication}\";\"{options.replication_power}\";\"{options.num_commons}\";\"{int(options.whales_stake_percentage)}\";\"{options.num_whales}\";"
    void_blocks = f"\"{void_blocks_percentage:.2f}\";"
    underliners_threshold = f"\"{underline_stake / total_staked * 100:.2f}\";"
    underliners_percentile = f"\"{num_underliners / num_stakers * 100:.2f}\";"
    whales_elegibility = f"\"{whales_mined_blocks / options.epochs * 100:.2f}\";"
    whales_latency = f"\"{whales_first_epoch}\";"
    try:
      with open(csv_filename, "a", encoding="utf-8") as csv_file:
        row = modus_ponens + input_params + void_blocks + underliners_threshold + underliners_percentile + whales_elegibility + whales_latency
        csv_file.write(row + '\n')
    except Exception as ex:
      return

if __name__ == "__main__":
    main()
