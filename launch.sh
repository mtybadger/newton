#!/usr/bin/env python3

import subprocess
import sys

# Launch physics.py for batches 0-7 (videos 0-799) in parallel threads
processes = []
for i in range(10):
    print(f"Starting batch {i} (videos {i*100}-{(i+1)*100-1})")
    process = subprocess.Popen(["/home/spruce/Downloads/blender/blender", "-b", "-P", "physics.py", "--", str(i)])
    processes.append(process)

# Wait for all processes to complete
for process in processes:
    process.wait()

print("All batches complete")
