#!/usr/bin/python3

import optparse
import random
import sys
import tqdm

import matplotlib.pyplot as plt

def build_stakers(num_stakers, largest_staker, stake_per_node, total_staked):
    stakers = [random.random() for s in range(int(num_stakers - largest_staker / stake_per_node))]
    stakers = [s / (sum(stakers) * (1 / (1 - largest_staker / total_staked))) for s in stakers]
    stakers.extend([stake_per_node / total_staked for i in range(int(largest_staker / stake_per_node))])
    return stakers

def simulate_data_request(stakers, witnesses, attacker_nodes, random_selection):
    # eligibility: vrf < 2 ^ 256 * own_power / global_power * witnesses * 2 ^ round
    others_eligibility = [0, 0, 0, 0]
    attacker_eligiblity = [0, 0, 0, 0]
    for i in range(0, 4):
        for j, staker in enumerate(stakers):
            # Eligibility is completely random:
            if random_selection:
                eligibility = 1 / len(stakers) * witnesses * (2 ** i)
            # Eligibility is proportional to the staker's balance
            else:
                eligibility = staker * witnesses * (2 ** i)

            if random.random() < eligibility:
                if j >= len(stakers) - attacker_nodes:
                    assert stakers[j] == 0.01
                    attacker_eligiblity[i] += 1
                else:
                    others_eligibility[i] += 1
    return others_eligibility, attacker_eligiblity

def main():
    parser = optparse.OptionParser()
    parser.add_option("--stakers", type="int", default=100, dest="stakers")
    parser.add_option("--largest-staker", type="int", default=10000000, dest="largest_staker")
    parser.add_option("--stake-per-node", type="int", default=1000000, dest="stake_per_node")
    parser.add_option("--total-staked", type="int", default=100000000, dest="total_staked")
    parser.add_option("--data-requests", type="int", default=100000, dest="data_requests")
    parser.add_option("--witnesses", type="int", default=10, dest="witnesses")
    parser.add_option("--random", action="store_true", dest="random")
    options, args = parser.parse_args()

    stakers = build_stakers(options.stakers, options.largest_staker, options.stake_per_node, options.total_staked)

    successful_data_requests = [0, 0, 0, 0]
    successful_data_request_attacks = [0, 0, 0, 0]
    attacker_nodes = int(options.largest_staker / options.stake_per_node)
    for epoch in tqdm.tqdm(range(options.data_requests)):
        others_eligibility, attacker_eligiblity = simulate_data_request(stakers, options.witnesses, attacker_nodes, options.random)
        for i, (oe, ae) in enumerate(zip(others_eligibility, attacker_eligiblity)):
            if oe >= options.witnesses:
                successful_data_requests[i] += 1
            if ae >= int(options.witnesses / 2) + 1:
                successful_data_request_attacks[i] += 1

    previous_success = 0
    for i, (so, sa) in enumerate(zip(successful_data_requests, successful_data_request_attacks)):
        success = so / options.data_requests
        print(f"Successful data requests @ commit round {i + 1}: {success * 100:.2f}%")
        successful_attack = sa / options.data_requests
        print(f"Successful data request attack @ commit round {i + 1}: {successful_attack * 100:.2f}%")
        dependent_successful_attack = (1 - previous_success) * successful_attack
        print(f"Dependent successful data request attacks @ commit round {i + 1}: {dependent_successful_attack * 100:.2f}%")
        print("")
        previous_success = success

if __name__ == "__main__":
    main()
