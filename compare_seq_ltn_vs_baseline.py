import subprocess
import os
import sys

runs = [
    {"script": "train_seq_baselines.py", "env": {"BACKBONE": "bilstm"}},
   # {"script": "train_seq_with_ltn.py", "env": {"BACKBONE": "bilstm"}},
    {"script": "train_seq_baselines.py", "env": {"BACKBONE": "cnn_bilstm"}},
   # {"script": "train_seq_with_ltn.py", "env": {"BACKBONE": "cnn_bilstm"}},
]

for run in runs:
    env = os.environ.copy()
    if run.get("env"):
        env.update(run["env"])
    print(f"Running {run['script']} with {run['env']}")
    subprocess.run([sys.executable, run["script"]], check=True, env=env)

print("All runs completed. Compare saved models in model/ and logs on stdout.") 