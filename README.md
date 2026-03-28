# AI-Powered RoboTaxi Simulation - COMP6032 Resit

Multi-agent taxi dispatch simulation implementing intelligent pathfinding and fare allocation algorithms.

## Key Improvements (Resit)

- **Fare Type Prediction**: KNN classifier (scikit-learn) to predict fare types (rich, budget, hurry, normal) for dynamic pricing
- **Crank Fare Detection**: Rejects suspicious fares (too short, too cheap, too expensive)
- **Faulty Taxi Tracking**: Identifies PsychoUber taxi by monitoring performance metrics
- **A* Pathfinding**: Traffic-aware optimal routing (0.2ms avg computation)
- **Multi-Factor Dispatcher**: Balances proximity, workload, capital, and fairness

## Results (Second Run)

- Fares Completed: **93** (from 597 generated)
- Cancellation Rate: **72.36%**
- Total Revenue: **£2143.80**
- Avg Revenue per Taxi: **£535.95**
- Avg Path Computation: **0.211 ms**

## Requirements

- Python 3.8+
- pygame 2.5.0+
- numpy 1.24.0+
- scikit-learn 1.0.0+

## Installation

```bash
pip install -r requirements.txt
```
