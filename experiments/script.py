import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_seconds', type=int, default=10)
args = parser.parse_args()

print(f"Server will run for {args.n_seconds} seconds")
for i in range(args.n_seconds):
    print("Running")
    time.sleep(1)

print("Stopped running")
