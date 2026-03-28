# AI-Powered RoboTaxi Simulation - COMP6032 Resit

Multi-agent taxi dispatch simulation implementing intelligent pathfinding and fare allocation algorithms.

## Key Improvements

- **A* Pathfinding**: Traffic-aware optimal routing for efficient navigation
- **Intelligent Bidding**: Workload-aware bidding with crank fare detection
- **Multi-Factor Dispatcher**: Balances proximity, workload, capital, and fairness
- **Dynamic Pricing**: Rule-based pricing based on distance, travel time, and congestion
- **Faulty Taxi Detection**: Identifies underperforming taxis
- 
## Results (4 Simulation Runs)

| Run | Fares Generated | Fares Completed | Fares Cancelled | Cancellation Rate | Total Revenue |
|-----|-----------------|-----------------|-----------------|-------------------|---------------|
| 1 (Baseline) | 541 | 44 | 414 | 76.52% | £776.52 |
| 2 (A*) | 574 | 118 | 385 | 67.07% | £2,167.74 |
| 3 (A*) | 550 | 111 | 373 | 67.82% | £2,008.44 |
| 4 (Dispatcher Fix) | 550 | 81 | 393 | 71.45% | £1,448.82 |
| **Average (Runs 2-4)** | **558** | **103** | **384** | **68.78%** | **£1,875.00** |

- **Avg Path Computation**: 0.225 ms
- **Faulty Taxi Detected**: Taxi 102 excluded after 10+ fares with >50% cancellation rate

- **Note**: Run 4 has lower completions because the dispatcher identified Taxi 102 as faulty and excluded it from allocations. This demonstrates the detection is working correctly, even though total revenue decreased.

## Key Improvements

- **A* Pathfinding**: Traffic-aware optimal routing using Manhattan heuristic (avg 0.225ms computation)
- **Intelligent Bidding**: Workload-aware bidding with crank fare detection
- **Multi-Factor Dispatcher**: Balances proximity, workload, capital, and fairness
- **Dynamic Pricing**: Rule-based pricing based on distance, travel time, and congestion
- **Faulty Taxi Detection**: Dispatcher identifies PsychoTaxi (Taxi 102) by monitoring cancellation rates (>50%) and excludes from allocations

## Task 3b: Ethical and Commercial Deployment Analysis

### Privacy Concerns
The system collects passenger origin, destination, and trip times. This data must be anonymized and stored securely. Passengers should have opt-out options.

### Algorithmic Fairness
The dispatcher's multi-factor scoring (proximity, workload, capital) could inadvertently discriminate if weights are biased. Further testing with diverse fare distributions is needed to ensure equitable service.

### Safety and Accountability
The faulty PsychoTaxi detection demonstrates the need for robust monitoring. In production, a human-in-the-loop would be required for safety-critical decisions.

### Economic Impact
Autonomous taxis will displace human drivers. A responsible deployment would include retraining programs and phased implementation.

### Transparency
Our rule-based pricing (distance, travel time, congestion) is explainable, unlike black-box ML. This builds passenger trust.


## Requirements

- Python 3.8+
- pygame 2.5.0+
- numpy 1.24.0+

## Installation

```bash
pip install -r requirements.txt
