## Tools to calculate different Proof-of-Stake metrics

### test_mining_eligibility.py
./test_mining_eligibility.py [--stakers int] [--distribution str] [--epochs int] [--replication int]

Accepted arguments for the staking distribution: gauss, uniform, beta, exponential, gamma

### test_witness_attack.py

./test_witness_attack.py [--staker int] [--largest-staker int] [--stake-per-node int] [--total-staked int] [--data-requests int] [--witnesses int] [--random]

random: use random eligibility for data request witnessing instead of stake ratio

### simulate_mining.py
./simulate_mining.py [options]

Options:

- --logging
    - explanation: set the logging level
    - default: info
    - allowed values: debug, info, warning, error, critical
- --epochs
    - explanation: the number of epochs for which to run the mining simulator
    - default: 100000
- --block-reward
    - explanation: set the block reward for an accepted block
    - default: 250
- --max-staking-txs-per-block
    - explanation: max number of staking transactions allowed per block
    - default: 32
- --mining-eligibility
    - explanation: allows setting different strategies for miner selection
    - default: vrf-stake-adaptative
    - allowed values: vrf-stake-adaptative, vrf-stake, module-stake, modulo-slot
- --coin-ageing
    - explanation: allows to experiment with different coin age strategies
    - default: reset
    - allowed values: reset, capped, halving
- --distribution
    - explanation: defines the distribution of the amount each staker (relative to the total amount)
    - default: random
    - allowed values: random, uniform, beta, exponential, gamma
- --replication-selector
    - explanation: sets a selection strategy when replication is enabled (vrf-stake and modulo-slot)
    - default: lowest-hash
    - allowed values: lowest-hash, highest-power
- --replication
    - explanation: set the replication factor to approximate the number of blocks generated per epoch
    - default: 16
- --replication-power
    - explanation: exponent to which the replication factor will be powered when delimiting maximum coin age
    - default: 2.0
- --stakers
    - explanation: number of initial stakers to simulate (before incoming whales)
    - default: 100
- --initial-staked
    - explanation: total amount intially staked (before incoming whales)
    - default: 1000000000
- --whales
    - explanation: adds a number of whale stakers to the initial staking distribution
    - default: 0
- --whales-stake-increment
    - explanation: percentage of initial stake to be added uniformly over the number of whales
    - default: 0

