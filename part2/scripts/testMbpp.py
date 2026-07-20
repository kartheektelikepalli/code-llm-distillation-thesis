from datasets import load_dataset

mbpp = load_dataset("mbpp")

problem = mbpp["test"][0]

print("Task ID:")
print(problem["task_id"])

print("\nPrompt:")
print(problem["text"])

print("\nReference Solution:")
print(problem["code"])

print("\nTests:")
print(problem["test_list"])