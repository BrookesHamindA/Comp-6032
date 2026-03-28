# AI-Powered RoboTaxi Simulation - COMP6032 Resit

Multi-agent taxi dispatch simulation implementing intelligent pathfinding and fare allocation algorithms.

## Key Improvements

- **A* Pathfinding**: Traffic-aware optimal routing for efficient navigation
- **Intelligent Bidding**: Workload-aware bidding with crank fare detection
- **Multi-Factor Dispatcher**: Balances proximity, workload, capital, and fairness
- **Dynamic Pricing**: Rule-based pricing based on distance, travel time, and congestion
- **Faulty Taxi Detection**: Identifies underperforming taxis
- 
## Results (3 Runs Final Summary)

| Run | Fares Generated | Fares Completed | Fares Cancelled | Cancellation Rate | Total Revenue |
|-----|-----------------|-----------------|-----------------|-------------------|---------------|
| 1 | 541 | 44 | 414 | 76.52% | £776.52 |
| 2 | 574 | 118 | 385 | 67.07% | £2,167.74 |
| 3 | 550 | 111 | 373 | 67.82% | £2,008.44 |
| **Average** | **555** | **91** | **391** | **70.47%** | **£1,650.90** |

- **Avg Path Computation**: 0.175 ms
- **Total Path Computations**: 926 across all runs

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
