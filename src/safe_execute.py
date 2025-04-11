import sys
import traceback
from io import StringIO
import threading
import multiprocessing

#
# def safe_exec(code):
#     # Redirect standard output
#     old_stdout = sys.stdout
#     sys.stdout = StringIO()
#
#     try:
#         # Using a helper function to encapsulate the exec call
#         def run_code():
#             exec_globals = {'__builtins__': __builtins__}
#             exec(code, exec_globals)
#
#         # Run the code in a safe environment
#         run_code()
#
#         # Retrieve the standard output content
#         output = sys.stdout.getvalue().strip()
#         return output if output else "Code executed successfully, but did not produce any output."
#     except Exception:
#         # Capture traceback and retrieve the last line with the error message
#         error_message = traceback.format_exc()
#         return f"An error occurred:{error_message}"
#     finally:
#         # Restore standard output
#         sys.stdout = old_stdout

class TimeoutException(Exception):
    pass

def run_code(conn, code):
    try:
        # Redirect standard output to capture print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        exec_globals = {'__builtins__': __builtins__}
        exec(code, exec_globals)

        # Send the captured output back through the pipe
        conn.send(sys.stdout.getvalue())
    except Exception as e:
        conn.send(f"An error occurred: {str(e)}")
    finally:
        sys.stdout = old_stdout
        conn.close()


def safe_exec(code, timeout=15):
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=run_code, args=(child_conn, code))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return "Code execution exceeded the time limit"
    else:
        if parent_conn.poll():
            return parent_conn.recv()
        return ""

# def safe_exec(code, timeout=2):
#     # Function to run the code
#     def run_code(output):
#         try:
#             exec_globals = {'__builtins__': __builtins__}
#             exec(code, exec_globals)
#         except Exception as e:
#             output.append(f"An error occurred: {str(e)}")
#
#     # Redirect standard output
#     old_stdout = sys.stdout
#     sys.stdout = StringIO()
#
#     output = []
#     thread = threading.Thread(target=run_code, args=(output,))
#     thread.start()
#     thread.join(timeout)
#
#     if thread.is_alive():
#         # Restore standard output
#         sys.stdout = old_stdout
#         return "Code execution exceeded the time limit"
#     else:
#         # Retrieve the standard output content
#         result = sys.stdout.getvalue().strip()
#         # Restore standard output
#         sys.stdout = old_stdout
#         if output:
#             return output[0]
#         return result if result else ""


# def safe_exec(code, timeout=2):
#     # Function to handle the timeout
#     def timeout_handler():
#         raise TimeoutException("Code execution exceeded the time limit")
#
#     # Redirect standard output
#     old_stdout = sys.stdout
#     sys.stdout = StringIO()
#
#     # Timer to enforce the timeout
#     timer = threading.Timer(timeout, timeout_handler)
#     timer.start()
#
#     try:
#         # Using a helper function to encapsulate the exec call
#         def run_code():
#             exec_globals = {'__builtins__': __builtins__}
#             exec(code, exec_globals)
#
#         # Run the code in a safe environment
#         run_code()
#
#         # Retrieve the standard output content
#         output = sys.stdout.getvalue().strip()
#         return output if output else ""
#     except TimeoutException as e:
#         return str(e)
#     except Exception:
#         # Capture traceback and retrieve the last line with the error message
#         error_message = traceback.format_exc()
#         return f"An error occurred: {error_message}"
#     finally:
#         # Restore standard output
#         sys.stdout = old_stdout
#         # Cancel the timer if the code execution finishes in time
#         timer.cancel()


# Example usage
if __name__ == '__main__':
    code = '''
    import itertools
    
    # Function to generate all possible permutations of the input numbers
    def get_permutations(numbers):
        return list(itertools.permutations(numbers))
    
    # Function to evaluate expressions
    def evaluate_expression(nums, ops):
        expression = ""
        for i in range(len(nums)):
            expression += str(nums[i])
            if i < len(ops):
                expression += ops[i]
        return expression
    
    # Perform brute-force search to find the expression that equals 24
    numbers = [2, 5, 5, 7]
    operators = ['+', '-', '*', '/']
    
    for perm in get_permutations(numbers):
        for ops in itertools.product(operators, repeat=3):
            for ops_parenthesis in itertools.product(['', '(', ')'], repeat=3):
                expression = evaluate_expression(perm, [op for pair in zip(ops_parenthesis, ops) for op in pair])
                try:
                    if eval(expression) == 24:
                        print("Expression: " + expression)
                except ZeroDivisionError:
                    pass
    '''

    code2 = """import itertools
# Given numbers
numbers = [3, 5, 11, 11]

# Generate all possible permutations of the numbers
perms = list(itertools.permutations(numbers))

# Iterate through each permutation and arithmetic operation to find the one resulting in 24
for perm in perms:
    for ops in itertools.product(['+', '-', '*', '/'], repeat=3):
        expression = f"{perm[0]} {ops[0]} {perm[1]} {ops[1]} {perm[2]} {ops[2]} {perm[3]}"
        try:
            if eval(expression) == 24:
                print("Answer:", expression)
                break
        except ZeroDivisionError:
            pass
    """

    code4 = """
def find_remainder():
    # Calculate a + b
    a_remainder = 64
    b_remainder = 99
    a = a_remainder  # since a = 70k + 64
    b = b_remainder  # since b = 105m + 99
    total = a + b
    
    # Find the remainder when total is divided by 35
    remainder = total % 35
    return remainder

print(find_remainder())

"""
    code5 = """
import time
time.sleep(3)
print('Hello, world!')
    """

    out = safe_exec(code4)
    print(out)
