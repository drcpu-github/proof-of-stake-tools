#!/usr/bin/python3

import datetime
import json
import logging
import numpy
import optparse
import os
import random
import sys
import time
import tqdm

from collections import Counter

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

    # Make whale stake also random, but constrained
    whale_stake = options.commons_staked * options.whales_stake_percentage / 100
    whales = [random.uniform(0.5, 2.0) for w in range(options.num_whales)]

    stakers = {
        i: commons[i] / sum(commons) * (options.commons_staked - options.num_commons * options.minimum_staked) + options.minimum_staked
            if i < options.num_commons
            else whales[i - options.num_commons] / sum(whales) * whale_stake
        for i in range(num_stakers)
    }

    return stakers

def simulate_eligibility_vrf_stake_linear(logger, epoch, data_request, stakers, coin_age, options, replication):
    logger.info(f"Simulating epoch {epoch + 1}, data request {data_request} (replication: {replication})")
    # eligibility: vrf < (2 ** 256) * own_power / global_power * rf

    # Create local copies with shorter variable names
    witnesses = options.data_requests_witnesses
    witnesses_selector = options.witnesses_selector

    witness_replication = witnesses * (2 ** replication)

    # Calculate global power
    powers = [min(stake * coin_age[staker], 147_573_952_589_676_416) for staker, stake in stakers.items()]
    max_power, num_stakers = max(powers), len(stakers)
    threshold_power = numpy.quantile(powers, 1 - witness_replication / num_stakers) if witness_replication < num_stakers else 0

    solvers = []
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = min(stake * coin_age[staker], 147_573_952_589_676_416)
        if own_power >= threshold_power:
            eligibility = own_power / ((replication / 4 * threshold_power + (4 - replication) / 4 * max_power) or 1)
        else:
            eligibility = 0
        vrf = random.random()
        if vrf < eligibility:
            logger.info(f"Staker {staker} is eligible to solve a data request: {vrf} < {eligibility}")
            # Miner with the lowest VRF value will be picked as the winner
            solvers.append((staker, own_power, vrf))

    if len(solvers) < witnesses:
        logger.warning(f"Not enough witnesses ({len(solvers)} < {witnesses}) found for data request {data_request}")
        return []
    else:
        logger.info(f"Found {len(solvers)} witnesses for data request {data_request}")
        if witnesses_selector == "lowest-vrf":
            solvers = sorted(solvers, key=lambda l: l[2])[:witnesses]
        elif witnesses_selector == "highest-power":
            solvers = sorted(solvers, key=lambda l: l[1], reverse=True)[:witnesses]
        logger.info(f"Stakers {solvers} are selected to solve a data request")
        return [s[0] for s in solvers]

def simulate_eligibility_vrf_stake_adaptative(logger, epoch, data_request, stakers, coin_age, options, replication):
    logger.info(f"Simulating epoch {epoch + 1}, data request {data_request} (replication: {replication})")
    # eligibility: vrf < (2 ** 256) * own_power / global_power * rf

    # Create local copies with shorter variable names
    witnesses = options.data_requests_witnesses
    witnesses_selector = options.witnesses_selector

    witness_replication = witnesses * (2 ** replication)

    # Calculate global power
    powers = [ min(stake * coin_age[staker], 147_573_952_589_676_416) for staker, stake in stakers.items() ]
    global_power = max(powers)
    num_stakers = len(stakers)
    threshold_power = numpy.quantile(powers, 1 - witness_replication / num_stakers) if witness_replication < num_stakers else 0

    logger.info(f"Threshold vs Global power: {threshold_power} vs {global_power} ({threshold_power / global_power * 100}%)")

    solvers = []
    for staker, stake in stakers.items():
        # Calculate power of the staker
        own_power = min(stake * coin_age[staker], 147_573_952_589_676_416)
        if options.witnesses_selector == "lowest-vrf":
            eligibility = own_power / global_power * replication if own_power >= threshold_power else 0
        else:
            eligibility = own_power / global_power if own_power >= threshold_power else 0
        vrf = random.random()
        if vrf < eligibility:
            logger.info(f"Staker {staker} is eligible to solve a data request: {vrf} < {eligibility}")
            # Miner with the lowest VRF value will be picked as the winner
            solvers.append((staker, own_power, vrf))

    if len(solvers) < witnesses:
        logger.warning(f"Not enough witnesses ({len(solvers)} < {witnesses}) found for data request {data_request}")
        return []
    else:
        logger.info(f"Found {len(solvers)} witnesses for data request {data_request}")
        if witnesses_selector == "lowest-vrf":
            solvers = sorted(solvers, key=lambda l: l[2])[:witnesses]
        elif witnesses_selector == "highest-power":
            solvers = sorted(solvers, key=lambda l: l[1], reverse=True)[:witnesses]
        logger.info(f"Stakers {solvers} are selected to solve a data request")
        return [s[0] for s in solvers]

def update_coin_age_reset(num_stakers, coin_age, solvers, options):
    for staker in range(num_stakers):
        if staker not in solvers:
            coin_age[staker] = coin_age[staker] + 1
        else:
            coin_age[staker] = 1
    return coin_age

def update_coin_age_collateral(stakers, coin_age, solvers, options):
    for staker in range(len(stakers)):
        if staker not in solvers:
            coin_age[staker] = coin_age[staker] + 1
        else:
            coin_age[staker] = (1 - solvers[staker] * options.data_requests_collateral / stakers[staker]) * coin_age[staker]
    return coin_age

def print_solver_stats(logger, stakers, solved_data_requests, total_data_requests):
    logger.info("Solver: data requests (percentage) -- stake (percentage)")
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=True):
        drs = solved_data_requests[staker] if staker in solved_data_requests else 0
        dr_percentage = drs / total_data_requests * 100
        stake_percentage = stake / sum(stakers.values()) * 100
        logger.info(f"{staker}: {drs} ({dr_percentage:.2f}%) -- {stake / 1E6:.2f}M ({stake_percentage:.2f}%)")

def plot_stakers(stakers, num_stakers, total_staked, options, timestamp):
    fig, ax = plt.subplots(1, 1)
    ax.hist([stake / 1E6 for stake in stakers.values()], bins=64)
    ax.set_xlabel("Staked (M)")
    ax.set_ylabel("Number of stakers")
    whales_staked = round(options.commons_staked * options.whales_stake_percentage / 100)
    plt.title(f"total stake = {round(total_staked / 1E6)}M, commons = {options.distribution}({num_stakers - options.num_whales}, {round((total_staked - whales_staked) / 1E6)}M), whales = uniform({options.num_whales}, {int(whales_staked/1E6)}M)")
    plt.savefig(f"plots/stakers_{plots_file_prefix(options)}_{timestamp}.png", bbox_inches="tight")

def plot_solving_rate(stakers, solved_data_requests, total_data_requests, options, plot_title, timestamp):
    dr_percentages, stake_percentages = [], []
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=True):
        drs = solved_data_requests[staker] if staker in solved_data_requests else 0
        dr_percentages.append(drs / total_data_requests * 100)
        stake_percentages.append(stake / sum(stakers.values()) * 100)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(stake_percentages, dr_percentages)
    ax.set_xlabel("Staked (%)")
    ax.set_ylabel("Data requests solved (%)")
    plt.title(plot_title)
    plt.savefig(f"plots/solving_{plots_file_prefix(options)}_{timestamp}.png", bbox_inches="tight")

def plot_data_requests_solved_at(data_requests_solved_at_attempt, options, plot_title, timestamp):
    x_values, y_values = [], []
    for x, y in data_requests_solved_at_attempt.items():
        x_values.append(x)
        y_values.append(y / sum(data_requests_solved_at_attempt.values()))

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(max(10, 0.5 * len(x_values)), 8)
    ax.bar(x_values, y_values)
    ax.xaxis.set_ticks(range(min(x_values), max(x_values) + 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(plot_title)
    plt.savefig(f"plots/drs_solved_{plots_file_prefix(options)}_{timestamp}.png", bbox_inches="tight")

def plots_file_prefix(options):
    return f"e={options.eligibility}_ca={options.coin_ageing}_dre={options.data_requests_per_epoch}_drd={options.data_requests_distribution}_drw={options.data_requests_witnesses}_ws={options.witnesses_selector}_d={options.distribution}_sc={str(options.num_commons).rjust(4, '0')}_swp={str(int(options.whales_stake_percentage)).rjust(2, '0')}_sw={str(options.num_whales).rjust(3, '0')}"

def main():
    parser = optparse.OptionParser()
    parser.add_option("--stakers", type="int", default=100, dest="num_commons")
    parser.add_option("--load-stakers", type="string", dest="load_stakers")
    parser.add_option("--dump-stakers", type="string", dest="dump_stakers")
    parser.add_option("--minimum-stake", type="int", default=10_000, dest="minimum_staked")
    parser.add_option("--initial-stake", type="int", default=1_000_000_000, dest="commons_staked")
    parser.add_option("--whales", type="int", default=0, dest="num_whales")
    parser.add_option("--whales-stake-increment", type="float", default=0, dest="whales_stake_percentage")
    parser.add_option("--distribution", type="string", default="random", dest="distribution")
    parser.add_option("--epochs", type="int", default=100000, dest="epochs")
    parser.add_option("--data-requests-per-epoch", type="int", default=3, dest="data_requests_per_epoch")
    parser.add_option("--data-requests-distribution", type="string", default="uniform", dest="data_requests_distribution")
    parser.add_option("--data-requests-witnesses", type="int", default=10, dest="data_requests_witnesses")
    parser.add_option("--data-requests-collateral", type="int", default=10, dest="data_requests_collateral")
    parser.add_option("--witnesses-selector", type="string", default="highest-power", dest="witnesses_selector")
    parser.add_option("--coin-ageing", type="string", default="reset", dest="coin_ageing")
    parser.add_option("--eligibility", type="string", default="vrf-stake-adaptative", dest="eligibility")
    parser.add_option("--logging", type="string", default="info", dest="logging")
    parser.add_option("--seed-randomness", action="store_true", dest="seed_randomness")
    options, args = parser.parse_args()

    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists("logs"):
        os.mkdir("logs")

    allowed_eligibility_strategies = ("vrf-stake-adaptative", "vrf-stake-linear")
    if options.eligibility not in allowed_eligibility_strategies:
        print(f"Unknown eligibility strategy: {', '.join(allowed_eligibility_strategies)}")
        sys.exit(1)
    allowed_data_requests_distributions = ("uniform", "random")
    if options.data_requests_distribution not in allowed_data_requests_distributions:
        print(f"Unknown data requests distribution: {', '.join(allowed_data_requests_distributions)}")
        sys.exit(1)
    allowed_witness_selectors = ("highest-power", "lowest-vrf")
    if options.witnesses_selector not in allowed_witness_selectors:
        print(f"Unknown witness selector: {', '.join(allowed_witness_selectors)}")
        sys.exit(1)
    allowed_coin_ageing = ("disabled", "reset", "collateral", )
    if options.coin_ageing not in allowed_coin_ageing:
        print(f"Unknown coin ageing strategy: {', '.join(allowed_coin_ageing)}")
        sys.exit(1)

    if options.data_requests_witnesses > (options.num_commons + options.num_whales) / 2:
        print(f"Amount of data request witnesses ({options.data_requests_witnesses}) cannot exceed half of the total stakers ({options.num_commons + options.num_whales})")
        sys.exit(1)

    if options.seed_randomness:
        random.seed(1337)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    short_timestamp = datetime.datetime.now().strftime("%H%M%S")

    logger = configure_logger("solving", timestamp, options.logging)

    logger.info(f"Command line arguments: {options}")

    if options.load_stakers:
        with open(options.load_stakers) as fh:
            contents = json.load(fh)
            options.num_commons = contents["num_commons"]
            options.minimum_staked = contents["minimum_staked"]
            options.commons_staked = contents["commons_staked"]
            options.num_whales = contents["num_whales"]
            options.whales_stake_percentage = contents["whales_stake_percentage"]
            options.distribution = contents["distribution"]
            stakers = {int(staker): float(amount) for staker, amount in contents["stakers"].items()}
    else:
        stakers = build_stakers(logger, options, timestamp)
    if options.dump_stakers:
        with open(options.dump_stakers, "w+") as fh:
            json.dump(
                {
                    "num_commons": options.num_commons,
                    "minimum_staked": options.minimum_staked,
                    "commons_staked": options.commons_staked,
                    "num_whales": options.num_whales,
                    "whales_stake_percentage": options.whales_stake_percentage,
                    "distribution": options.distribution,
                    "stakers": stakers,
                },
                fh,
            )
    logger.info(f"Stakers: {stakers}")
    plot_stakers(stakers, len(stakers), sum(stakers.values()), options, timestamp)

    num_stakers = len(stakers)
    coin_age = {staker: 1 for staker in range(num_stakers)}
    logger.info(f"Initial coin age: {coin_age}")

    data_requests_solved_at_attempt = {}
    solved_requests, data_requests_this_epoch, data_requests_per_epoch = {}, {}, {}
    failed_data_requests, data_requests_created = 0, 0
    potential_whale_manipulation_majority, potential_whale_manipulation_super_majority = 0, 0
    for epoch in tqdm.tqdm(range(options.epochs)):
        # Generate a number of data requests
        if options.data_requests_distribution == "uniform":
            num_data_requests = options.data_requests_per_epoch
        else:
            num_data_requests = random.randint(0, options.data_requests_per_epoch)
        if num_data_requests not in data_requests_per_epoch:
            data_requests_per_epoch[num_data_requests] = 0
        data_requests_per_epoch[num_data_requests] += 1

        # Simulate the data requests
        solvers = []
        data_requests_this_epoch[1] = [data_requests_created + d + 1 for d in range(num_data_requests)]
        data_requests_left = {k: list(v) for k, v in data_requests_this_epoch.items()}
        for attempt, data_requests in sorted(data_requests_this_epoch.items()):
            for dr in data_requests:
                if options.eligibility == "vrf-stake-adaptative":
                    dr_solvers = simulate_eligibility_vrf_stake_adaptative(logger, epoch, dr, stakers, coin_age, options, attempt)
                elif options.eligibility == "vrf-stake-linear":
                    dr_solvers = simulate_eligibility_vrf_stake_linear(logger, epoch, dr, stakers, coin_age, options, attempt)

                data_requests_left[attempt].remove(dr)
                if dr_solvers == []:
                    if attempt + 1 > 4:
                        logger.warning(f"Failed to resolve data request {dr}")
                        failed_data_requests += 1
                        continue
                    if attempt + 1 not in data_requests_left:
                        data_requests_left[attempt + 1] = []
                    data_requests_left[attempt + 1].append(dr)
                else:
                    if attempt not in data_requests_solved_at_attempt:
                        data_requests_solved_at_attempt[attempt] = 0
                    data_requests_solved_at_attempt[attempt] += 1
                    solvers.extend(dr_solvers)

                    whale_solvers = 0
                    for solver in dr_solvers:
                        if solver >= options.num_commons:
                            whale_solvers += 1
                    if whale_solvers > options.data_requests_witnesses / 2:
                        potential_whale_manipulation_majority += 1
                    if whale_solvers >= 7 * options.data_requests_witnesses / 10:
                        potential_whale_manipulation_super_majority += 1

            if attempt == 1:
                data_requests_created += num_data_requests

        assert len(data_requests_left[1]) == 0, ', '.join(f'{key}: {value}' for key, value in sorted(data_requests_left.items()))
        data_requests_this_epoch = {k: v for k,v in data_requests_left.items()}

        for solver in solvers:
            if solver not in solved_requests:
                solved_requests[solver] = 0
            solved_requests[solver] += 1

        # No data requests are solved, don't update coin age
        if len(solvers) == 0:
            continue

        if options.coin_ageing == "disabled":
            pass
        elif options.coin_ageing == "reset":
            coin_age = update_coin_age_reset(num_stakers, coin_age, Counter(solvers), options)
        elif options.coin_ageing == "collateral":
            coin_age = update_coin_age_collateral(stakers, coin_age, Counter(solvers), options)
        logger.info(f"Updated coin age: {coin_age}")

    total_data_requests = sum(data_requests * amount for data_requests, amount in data_requests_per_epoch.items())

    print_solver_stats(logger, stakers, solved_requests, total_data_requests)

    whales_staked = options.commons_staked * options.whales_stake_percentage / 100
    total_staked = options.commons_staked + whales_staked

    whales_str = ""
    if options.num_whales > 0:
        whales_str = f", whales = {options.num_whales} ({int(whales_staked / 1E6)}M)"
    plot_title = f"commons = {options.num_commons} ({int(options.commons_staked / 1E6)}M){whales_str}, witnesses selector = {options.witnesses_selector}, coin ageing = {options.coin_ageing}"
    plot_solving_rate(stakers, solved_requests, total_data_requests, options, plot_title, short_timestamp)

    data_requests_solved_at_attempt[-1] = failed_data_requests
    plot_data_requests_solved_at(data_requests_solved_at_attempt, options, plot_title, short_timestamp)

    print("\nSimulation parameters:")
    print("> Number of epochs:                    ", options.epochs)
    print("> Mining eligiblity:                   ", options.eligibility)
    print("> Coin ageing:                         ", options.coin_ageing)
    print("> Commons distribution:                ", options.distribution)
    print("> Data request per epoch:              ", options.data_requests_per_epoch)
    print("> Data request witnesses:              ", options.data_requests_witnesses)
    print("> Witnesses selector:                  ", options.witnesses_selector)
    print("> Commons initial stake:               ", f"{int(options.commons_staked/1E6):,} MWIT (÷ {options.num_commons:,} nodes)")
    if (options.num_whales > 0):
        print("> Whales incoming stake:               ", f"+{whales_staked / options.commons_staked * 100:.2f}% => {int(total_staked/1E6):,} MWIT")
        percentage_str = f"{whales_staked / total_staked * 100:.2f}"
        print("> Whales unitary stake:                ", f"{int(whales_staked/options.num_whales):,} WIT (x {options.num_whales:,} nodes)")

    print(f"\nSimulation results:")

    failed_data_requests_percentage = failed_data_requests / total_data_requests * 100
    percentage_str = f"{failed_data_requests_percentage:.2f}"
    print(f"> Total data requests:                  {total_data_requests:,} data requests")
    print(f"> Failed data requests:                 {failed_data_requests:,} ({percentage_str} %)")
    percentage_str = f"{potential_whale_manipulation_majority / total_data_requests * 100:.2f}"
    print(f"> Manipulatable data requests (>50%):   {potential_whale_manipulation_majority:,} ({percentage_str} %)")
    percentage_str = f"{potential_whale_manipulation_super_majority / total_data_requests * 100:.2f}"
    print(f"> Manipulatable data requests (>=70%):  {potential_whale_manipulation_super_majority:,} ({percentage_str} %)")

    underline_stake, num_underliners = 0, 0
    for staker, stake in sorted(stakers.items(), key=lambda l: l[1], reverse=False):
        if staker not in solved_requests or solved_requests[staker] / options.epochs <= stake / total_staked / 10:
            num_underliners += 1
            if stake > underline_stake:
                underline_stake = stake

    percentage_str = f"{underline_stake / total_staked * 100:.2f}"
    print(f"> Underliners threshold:                {percentage_str} % ({int(underline_stake):,} WIT)")
    percentage_str = f"{num_underliners / num_stakers * 100:.2f}"
    print(f"> Underliners percentile:               {percentage_str} % ({num_underliners:,} nodes)")

    if options.num_whales > 0:
        whales_data_requests_solved = 0
        for staker, num_solved in solved_requests.items():
            if staker >= options.num_commons:
                whales_data_requests_solved += num_solved

        percentage_str = f"{whales_staked / total_staked * 100:.2f}"
        print(f"> Whales relative stake:                {percentage_str} % ({options.num_whales / num_stakers * 100:.2f}% of nodes)")
        percentage_str = f"{whales_data_requests_solved / total_data_requests / options.num_whales * 100:.2f}"
        print(f"> Whales eligibility:                   {percentage_str} %")
        save_simulation_results(short_timestamp, options, num_stakers, total_staked, failed_data_requests_percentage, underline_stake, num_underliners, whales_data_requests_solved)
    else:
        save_simulation_results(short_timestamp, options, num_stakers, total_staked, failed_data_requests_percentage, underline_stake, num_underliners, 0)

def save_simulation_results(index, options, num_stakers, total_staked, underline_stake, failed_data_requests_percentage, num_underliners, whales_data_requests_solved):
    csv_filename = f"results/{options.eligibility}_{options.coin_ageing}_{options.witnesses_selector}_{options.distribution}.csv"
    modus_ponens = f"\"{options.eligibility}\";\"{options.coin_ageing}\";\"{options.witnesses_selector}\";\"{options.distribution}\";"
    input_params = f"\"{index}\";\"{options.num_commons}\";\"{int(options.whales_stake_percentage)}\";\"{options.num_whales}\";"
    failed_data_requests = f"\"{failed_data_requests_percentage:.2f}\";"
    underliners_threshold = f"\"{underline_stake / total_staked * 100:.2f}\";"
    underliners_percentile = f"\"{num_underliners / num_stakers * 100:.2f}\";"
    whales_eligibility = f"\"{whales_data_requests_solved / options.epochs * 100:.2f}\";"
    try:
      with open(csv_filename, "a", encoding="utf-8") as csv_file:
        row = modus_ponens + input_params + failed_data_requests + underliners_threshold + underliners_percentile + whales_eligibility
        csv_file.write(row + '\n')
    except Exception as ex:
      return

if __name__ == "__main__":
    main()
