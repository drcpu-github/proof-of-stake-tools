## Tools to calculate different Proof-of-Stake metrics

### test_mining_eligibility.py
./test_mining_eligibility.py [--stakers int] [--distribution str] [--epochs int] [--replication int]

Accepted arguments for the staking distribution: gauss, uniform, beta, exponential, gamma

### test_witness_attack.py

./test_witness_attack.py [--staker int] [--largest-staker int] [--stake-per-node int] [--total-staked int] [--data-requests int] [--witnesses int] [--random]

random: use random eligibility for data request witnessing instead of stake ratio
