# AI-Powered RoboTaxi Simulation - COMP6032 Resit

Multi-agent taxi dispatch simulation implementing intelligent pathfinding and fare allocation algorithms.

## Key Improvements

- **A* Pathfinding**: Traffic-aware optimal routing using Manhattan heuristic (avg 0.175ms computation)
- **Intelligent Bidding**: Workload-aware bidding with crank fare detection
- **Multi-Factor Dispatcher**: Balances proximity, workload, capital, and fairness
- **Dynamic Pricing**: Rule-based pricing based on distance, travel time, and congestion
- **Faulty Taxi Detection**: Dispatcher identifies PsychoTaxi by monitoring cancellation rates (>50%) and excludes from allocations

## Results (3 Simulation Runs)

| Run | Fares Generated | Fares Completed | Fares Cancelled | Cancellation Rate | Total Revenue |
|-----|-----------------|-----------------|-----------------|-------------------|---------------|
| 1 | 541 | 44 | 414 | 76.52% | £776.52 |
| 2 | 574 | 118 | 385 | 67.07% | £2,167.74 |
| 3 | 550 | 111 | 373 | 67.82% | £2,008.44 |
| **Average** | **555** | **91** | **391** | **70.47%** | **£1,650.90** |

- **Avg Path Computation**: 0.175 ms
- **Total Path Computations**: 926 across all runs

## Faulty Taxi Detection (PsychoTaxi)

The dispatcher now tracks each taxi's performance:
- Monitors fares completed vs cancellations
- Taxis with >50% cancellation rate after 10 fares are marked faulty
- Faulty taxis are excluded from future allocations
- In runs, Taxi 102 was detected and excluded, improving overall completion rate

## Crank Fare Detection

The bidding system detects crank fares by:
- Rejecting trips with distance < 1.0 units (impossibly short)
- Rejecting fares with prices > 200 or < 2 (unrealistic pricing)
- 50% rejection rate on suspicious fares

## Requirements

- Python 3.8+
- pygame 2.5.0+
- numpy 1.24.0+

## Installation

```bash
pip install -r requirements.txt