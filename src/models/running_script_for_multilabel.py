import time
import os
import psutil
from ppi_multilabel_classification import main

start_time = time.time()
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)

main()

mem_after = process.memory_info().rss / (1024 * 1024)
end_time = time.time()

print("\n" + "=" * 60)
print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Memory before: {mem_before:.2f} MB")
print(f"Memory after:  {mem_after:.2f} MB")
print("=" * 60)
