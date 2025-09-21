from datetime import datetime
import os

class EvaluationFramework:
    def __init__(self, adversary, baseline):
        self.adversary = adversary
        self.baseline = baseline
        self.results = []

    def run_simulation(self, num_trials):
        for trial in range(num_trials):
            result = self.adversary.run_attack()
            self.results.append(result)
            self.log_result(trial, result)

    def log_result(self, trial, result):
        log_entry = f"{datetime.now()}: Trial {trial} - Result: {result}\n"
        with open("evaluation_log.txt", "a") as log_file:
            log_file.write(log_entry)

    def compare_performance(self):
        adversary_success = sum(result['success'] for result in self.results) / len(self.results)
        baseline_success = self.baseline.get_success_rate()
        comparison = {
            "adversary_success": adversary_success,
            "baseline_success": baseline_success,
            "improvement": adversary_success - baseline_success
        }
        return comparison

    def report_results(self):
        report_path = "evaluation_report.txt"
        with open(report_path, "w") as report_file:
            for result in self.results:
                report_file.write(f"Result: {result}\n")
            comparison = self.compare_performance()
            report_file.write(f"Comparison: {comparison}\n")