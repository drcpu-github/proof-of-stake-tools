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

- --stakers
    - explanation: number of total stakers to simulate
    - default: 100
- --total-staked
    - explanation: total amount staked
    - default: 100000000
- --distribution
    - explanation: defines the distribution of the amount each staker (relative to the total amount)
    - default: random
    - allowed values: random, uniform, beta, exponential, gamma
- --whales
    - explanation: adds a number of whale stakers to the staking distribution
    - default: 0
- --whales-stake
    - explanation: percentage of the total staked to divide uniformly over the number of whales
    - default: 0
- --epochs
    - explanation: the number of epochs for which to run the mining simulator
    - default: 100000
- --replication
    - explanation: set the replication factor to approximate the number of blocks generated per epoch
    - default: 16
- --coin-ageing
    - explanation: allows to experiment with different coin age strategies
    - default: reset
    - allowed values: reset, capped, halving
- --mining-eligibility
    - explanation: allows setting different strategies for miner selection
    - default: vrf-stake
    - allowed values: vrf-stake, module-stake
- --logging
    - explanation: set the logging level
    - default: info
    - allowed values: debug, info, warning, error, critical
