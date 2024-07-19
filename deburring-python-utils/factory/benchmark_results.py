class BenchmarkResult:
    """
    A class for storing and printing the results of deburring benchmark trials.

    Attributes:
    trial_results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains the results for a trial.
    """

    def __init__(self):
        """Initialize an empty BenchmarkResult object."""
        self.trial_results = []

    def add_trial_result(
        self,
        trial_id,
        trial_name,
        test_details,
    ):
        """Add the results for a trial to the BenchmarkResult object.

        The function updates the trial_results attribute by appending a new dictionary with the results for the trial.
        It computes the number of catastrophic_failures, failures and successes for the given trial.

        Args:
            trial_id : The ID of the trial.
            trial_name : The name of the trial.
            test_details : A list of dictionaries, where each dictionary contains the details for a test in the trial.
        """
        catastrophic_failures = 0
        failures = 0
        successes = 0
        avg_precision = 0
        avg_time = 0

        for test_detail in test_details:
            if test_detail["limits"] is not False:
                catastrophic_failures += 1
            elif test_detail["reach_time"] is not None:
                successes += 1
                avg_precision += test_detail["error_placement_tool"]
                avg_time += test_detail["reach_time"]
            else:
                failures += 1

        if successes > 0:
            avg_time = avg_time / successes
            avg_precision = avg_precision / successes
        else:
            avg_time = None
            avg_precision = None

        trial_result = {
            "trial_id": trial_id,
            "trial_name": trial_name,
            "catastrophic_failures": catastrophic_failures,
            "failures": failures,
            "successes": successes,
            "avg_reach_time": avg_time,
            "avg_precision": avg_precision,
            "test_details": test_details,  # List of dictionaries, each containing details about a test
        }
        self.trial_results.append(trial_result)

    def print_results(self, print_details=False):
        """Print the results for all trials stored in the BenchmarkResult object.

        Args:
            print_details: If True, print the details for each test in a trial. Defaults to False.
        """
        for trial_result in self.trial_results:
            print(f"Trial {trial_result['trial_id'] + 1}: {trial_result['trial_name']}")
            print(f"Catastrophic failures: {trial_result['catastrophic_failures']}")
            print(f"Failures: {trial_result['failures']}")
            print(f"Successes: {trial_result['successes']}")
            print(
                f"Average reach time for successes: {trial_result['avg_reach_time']}s",
            )
            print(
                f"Average precision for successes: {trial_result['avg_precision']}s",
            )
            if print_details:
                print("\nTest details:")
                for test_detail in trial_result["test_details"]:
                    print(test_detail)
            print("\n-------------------------\n")
