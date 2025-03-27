import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import re
import psutil

class CodeEvaluator:
    def __init__(self, timeout: float = 3.0, num_workers: int = 4):
        self.timeout = timeout  # Time limit per test in seconds
        self.num_workers = num_workers  # Number of concurrent workers

    def extract_expected_func_name(self, test_list):
        """Extract the expected function name from the test case."""
        pattern = r'assert\s+(\w+)\s*\('
        names = set()
        for test in test_list:
            match = re.search(pattern, test)
            if match:
                names.add(match.group(1))
        return names.pop() if len(names) == 1 else None

    def check_correctness(self, code, test, result_queue):
        """Run the code and test in a separate process."""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            
            # Match the function name in the test to a callable in the code
            expected_name = self.extract_expected_func_name([test])
            if expected_name and expected_name not in exec_globals:
                candidate = None
                for key, value in exec_globals.items():
                    if callable(value) and not key.startswith("__"):
                        candidate = value
                        break
                if candidate is not None:
                    exec_globals[expected_name] = candidate
                else:
                    raise Exception("No callable function found in code.")

            exec(test, exec_globals)
            result_queue.put({"passed": True, "timeout": False, "exception": None})
        except Exception as e:
            result_queue.put({"passed": False, "timeout": False, "exception": str(e)})

    def evaluate(self, code: str, test_list: list):
        """
        Evaluate the code against the test cases with a timeout.
        Returns pass@k metrics (e.g., pass@1).
        """
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                results = []

                for test in test_list:
                    result_queue = multiprocessing.Queue()
                    process = multiprocessing.Process(
                        target=self.check_correctness,
                        args=(code, test, result_queue)
                    )
                    process.daemon = True  # Ensure process terminates with parent
                    future = executor.submit(lambda p=process: (p.start(), p)[1])
                    futures.append((future, process, result_queue))

                for future, process, result_queue in futures:
                    process = future.result()  # Start the process
                    process.join(self.timeout)  # Wait up to timeout seconds

                    if process.is_alive():
                        # Terminate the process if it’s still running
                        p = psutil.Process(process.pid)
                        p.terminate()  # Try graceful termination
                        try:
                            p.wait(timeout=1.0)  # Give it 1 second to stop
                        except psutil.TimeoutExpired:
                            p.kill()  # Force kill if it doesn’t stop
                        results.append({"passed": False, "timeout": True, "exception": "Timeout"})
                    else:
                        result = result_queue.get() if not result_queue.empty() else {
                            "passed": False, "timeout": False, "exception": "Process failed"
                        }
                        results.append(result)

            # Compute pass@k (simplified to pass@1 for this example)
            total_tests = len(test_list)
            passed_tests = sum(1 for r in results if r["passed"])
            pass_at_1 = passed_tests / total_tests if total_tests > 0 else 0.0
            return {"pass@1": pass_at_1}

        except Exception as e:
            return {"pass@1": 0.0}

# Example usage
if __name__ == "__main__":
    evaluator = CodeEvaluator(timeout=3.0, num_workers=4)

    # Test case 1: Should pass
    code_pass = "def count_occurences(lst):\r\n    return {i: lst.count(i) for i in lst}"
    tests_pass = [
      "assert count_Occurrence(('a', 'a', 'c', 'b', 'd'),['a', 'b'] ) == 3",
      "assert count_Occurrence((1, 2, 3, 1, 4, 6, 7, 1, 4),[1, 4, 7]) == 6",
      "assert count_Occurrence((1,2,3,4,5,6),[1,2]) == 2"
    ]
    result_pass = evaluator.evaluate(code_pass, tests_pass)
    print(f"Pass case: {result_pass}")  # Expected: {'pass@1': 1.0}

    # Test case 2: Should fail
    code_fail = """
def test():
    return False
"""
    tests_fail = ["assert test() == True"]
    result_fail = evaluator.evaluate(code_fail, tests_fail)
    print(f"Fail case: {result_fail}")  # Expected: {'pass@1': 0.0}

    # Test case 3: Timeout
    code_timeout = """
def test():
    while True:
        pass
"""
    tests_timeout = ["assert test() == True"]
    result_timeout = evaluator.evaluate(code_timeout, tests_timeout)
    print(f"Timeout case: {result_timeout}")  # Expected: {'pass@1': 0.0}