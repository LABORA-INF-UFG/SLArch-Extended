import subprocess
import logging
import pandas as pd
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PRE = ROOT / "pre-process"
SPLIT = ROOT / "split"
POST = ROOT / "post-process"
FEEDBACK_DIR = ROOT / "feedback"

CSV_FILE = POST / "ue_complete_select_params.csv"

NS3_DIR = Path("/home/ufpel/anaconda3/envs/ns3-sl-fa/ns-3-dev")
NS3_CMD = [str(NS3_DIR / "ns3"), "run", "scratch/ns3_5glena_parallel_batch.cc"]

STEP1_CMD = ["python3", str(PRE / "step_1_net_fetch_compile_sort2.py")]
STEP2_CMD = ["python3", str(PRE / "step_2_net_generate_combined_stats.py")]
STEP3_CMD = ["python3", str(PRE / "step_3_emd_dataset_integration.py")]
STEP4_CMD = ["python3", str(PRE / "step_4_merge_files.py")]          
NET_SELECT_CMD = ["python3", str(POST / "net_select_agent2.py")]
SL_TRAIN_CMD = ["python3", str(SPLIT / "sl_parallel_batch2.py")]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("COORDINATOR")

def run_process(command, name, working_dir=None):
    log.info(f"Starting: {name}")
    process = subprocess.Popen(
        command,
        cwd=working_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        log.error(f"{name} failed")
        sys.exit(1)
    log.info(f"{name} completed successfully")

def extract_feedback():
    feedback_file = FEEDBACK_DIR / "feedback.json"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        log.warning("Final CSV missing – cannot extract feedback.")
        return

    try:
        df = pd.read_csv(CSV_FILE, sep=',')           
        latest = df.iloc[-1]

        feedback = {
            "avg_loss": latest.get('avg_loss', 0.0),
            "low_accuracy_clients": latest.get('low_accuracy_clients', []),
        }

        with open(feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)

        log.info(f"Feedback saved to {feedback_file}")

    except Exception as e:
        log.error(f"Failed to extract feedback: {e}")

def run_pipeline():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    run_process(NS3_CMD, "NS‑3 5G-LENA Simulation", NS3_DIR)    
    run_process(STEP1_CMD, "Fetch & compile network results")
    run_process(STEP2_CMD, "Generate combined network statistics")
    run_process(STEP3_CMD, "EMD and Datasets integration")
    print("")
    run_process(STEP4_CMD, "Merge network statistics with EMD parameters")
    print("\nRunning Network - Datasets parameters merging...\n")
    run_process(NET_SELECT_CMD, "Running Network Selection...")
    run_process(SL_TRAIN_CMD, "Split learning training")

    extract_feedback()

if __name__ == "__main__":
    print("")
    log.info("Starting Experiment Running")
    run_pipeline()
    log.info("Experiment finished running\n")
