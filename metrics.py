"""
Performance metrics collection for RoboTaxi simulation.
Tracks only essential client-required metrics: revenue, cancellations, computation time.
Only saves data for simulations > 60 minutes.
"""

import csv
import time
from datetime import datetime


class MetricsCollector:
    """Collects essential performance metrics for RoboTaxi simulation"""

    def __init__(self, run_id=1, algorithm_type="baseline", simulation_minutes=1440):
        self.run_id = run_id
        self.algorithm_type = algorithm_type
        self.simulation_minutes = simulation_minutes
        self.start_time = time.time()

        self.metrics = {
            "fares_generated": 0,
            "fares_completed": 0,
            "fares_cancelled": 0,
            "total_revenue": 0.0,
            "taxi_revenues": {},
            "path_computation_times": [],
        }

    def record_fare_generated(self):
        """Record when a new fare is generated"""
        self.metrics["fares_generated"] += 1

    def record_fare_completed(self, taxi_id, revenue):
        """Record successful fare completion"""
        self.metrics["fares_completed"] += 1
        self.metrics["total_revenue"] += revenue

        if taxi_id not in self.metrics["taxi_revenues"]:
            self.metrics["taxi_revenues"][taxi_id] = 0.0
        self.metrics["taxi_revenues"][taxi_id] += revenue

    def record_fare_cancelled(self):
        """Record fare cancellation"""
        self.metrics["fares_cancelled"] += 1

    def record_path_computation(self, computation_time_seconds):
        """Record pathfinding computation time in seconds"""
        self.metrics["path_computation_times"].append(computation_time_seconds)

    def get_cancellation_rate(self):
        """Calculate cancellation rate as percentage"""
        if self.metrics["fares_generated"] == 0:
            return 0.0
        return self.metrics["fares_cancelled"] / self.metrics["fares_generated"] * 100

    def get_avg_revenue_per_taxi(self):
        """Calculate average revenue per taxi"""
        if len(self.metrics["taxi_revenues"]) == 0:
            return 0.0
        return self.metrics["total_revenue"] / len(self.metrics["taxi_revenues"])

    def get_avg_computation_time_ms(self):
        """Calculate average path computation time in milliseconds"""
        if len(self.metrics["path_computation_times"]) == 0:
            return 0.0
        return (
            sum(self.metrics["path_computation_times"])
            / len(self.metrics["path_computation_times"])
        ) * 1000

    def finalize(self):
        """Calculate final metrics"""
        self.metrics["runtime_seconds"] = time.time() - self.start_time
        self.metrics["cancellation_rate"] = self.get_cancellation_rate()
        self.metrics["avg_revenue_per_taxi"] = self.get_avg_revenue_per_taxi()
        self.metrics["avg_computation_time_ms"] = self.get_avg_computation_time_ms()

    def print_summary(self):
        """Print metrics summary to console"""
        print("\n" + "=" * 70)
        print(
            f"SIMULATION METRICS - Run {self.run_id} ({self.algorithm_type}) - {self.simulation_minutes} min"
        )
        print("=" * 70)
        print(f"Fares Generated:          {self.metrics['fares_generated']}")
        print(f"Fares Completed:          {self.metrics['fares_completed']}")
        print(f"Fares Cancelled:          {self.metrics['fares_cancelled']}")
        print(f"Cancellation Rate:        {self.metrics['cancellation_rate']:.2f}%")
        print(f"Total Revenue:            £{self.metrics['total_revenue']:.2f}")
        print(f"Avg Revenue per Taxi:     £{self.metrics['avg_revenue_per_taxi']:.2f}")
        if len(self.metrics["path_computation_times"]) > 0:
            print(
                f"Avg Path Computation:     {self.metrics['avg_computation_time_ms']:.3f} ms"
            )
            print(
                f"Path Computations:        {len(self.metrics['path_computation_times'])}"
            )
        print(
            f"Actual Runtime:           {self.metrics['runtime_seconds']:.2f} seconds"
        )
        print("=" * 70 + "\n")

    def save_to_csv(self, filename="simulation_results.csv"):
        """
        Save metrics to CSV file.
        Only saves if simulation runtime is > 60 minutes.
        CSV contains only client-required fields.
        """
        if self.simulation_minutes <= 60:
            print(
                f"⚠ Simulation runtime ({self.simulation_minutes} min) ≤ 60 min - NOT saving to CSV"
            )
            return

        import os

        file_exists = os.path.exists(filename)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(
                    [
                        "run_id",
                        "algorithm",
                        "timestamp",
                        "simulation_minutes",
                        "fares_generated",
                        "fares_completed",
                        "fares_cancelled",
                        "cancellation_rate_%",
                        "total_revenue_gbp",
                        "avg_revenue_per_taxi_gbp",
                        "avg_path_computation_ms",
                        "num_path_computations",
                    ]
                )

            writer.writerow(
                [
                    self.run_id,
                    self.algorithm_type,
                    datetime.now().isoformat(),
                    self.simulation_minutes,
                    self.metrics["fares_generated"],
                    self.metrics["fares_completed"],
                    self.metrics["fares_cancelled"],
                    round(self.metrics["cancellation_rate"], 2),
                    round(self.metrics["total_revenue"], 2),
                    round(self.metrics["avg_revenue_per_taxi"], 2),
                    round(self.metrics["avg_computation_time_ms"], 3),
                    len(self.metrics["path_computation_times"]),
                ]
            )

        print(f"✓ Metrics saved to {filename}")
