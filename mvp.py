"""
Industrial AI Workload Management Platform — Demo with REAL Data
---------------------------------------------------------------------
Uses the real AI4I 2020 Predictive Maintenance Dataset (10,000 actual
industrial machine sensor readings: air/process temperature, rotational
speed, torque, tool wear, and real failure labels).
Source (public mirror): https://raw.githubusercontent.com/michele-abruzzese/predictive_maintenance/main/predictive_maintenance.csv
(Original dataset: UCI Machine Learning Repository / commonly hosted on Kaggle)

 Workload (Process)	    What it represents in the plant	What it actually computes on the real data
Failure-Detection-AI	  An AI service that watches machines for signs of failure	Scans its 2,500 machine records, counts how many actually failed, and breaks failures down by type (Power Failure, Overstrain, Tool Wear, etc.)
Temperature-Monitor-AI	A thermal-monitoring service for equipment safety	Computes min/max/average air & process temperature across its machines, and logs every machine's readings
Torque-Analysis-AI    	A mechanical-stress monitoring service	Computes average torque & rotational speed, and flags machines running above a safe torque threshold (overload risk)
Tool-Wear-Risk-AI	      A predictive-maintenance service	Computes average/max tool wear and flags machines that have worn past a threshold and likely need servicing soon

What this script does:
  1. Downloads the real CSV (or reuses it if already downloaded) and
     splits it into 4 chunks -- one per "industrial AI workload".
  2. Each workload is a REAL process that does real analysis on its
     chunk of real machine data (failure counting, torque/temperature
     stats, tool-wear risk scoring, per-failure-type breakdown) and
     writes the results to a text file.
  3. Evaluates FCFS, SJF, Priority, and Round Robin scheduling (on
     paper, using estimated burst times) and picks the one with the
     lowest average waiting time.
  4. Actually executes the workloads for real, concurrently, in the
     order the winning algorithm chose, synchronized with a Lock.
  5. Prints final statistics: scheduling stats + real execution time +
     the actual analysis results pulled from real data.

Run:  python workload_demo.py
Output text files are written to ./workload_outputs/
"""

import os
import csv
import time
import threading
import urllib.request
from dataclasses import dataclass, field
from queue import Queue

DATA_URL = "https://raw.githubusercontent.com/michele-abruzzese/predictive_maintenance/main/predictive_maintenance.csv"
DATA_FILE = "predictive_maintenance.csv"
OUTPUT_DIR = "workload_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 0. Fetch the real dataset (once) and split it into 4 chunks
# ---------------------------------------------------------------------------
def load_real_dataset():
    if not os.path.exists(DATA_FILE):
        print(f"[Data] Downloading real dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    with open(DATA_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"[Data] Loaded {len(rows)} real machine sensor readings "
          f"(AI4I 2020 Predictive Maintenance Dataset)")
    return rows


def split_into_chunks(rows, n=4):
    size = len(rows) // n
    return [rows[i * size: (i + 1) * size] for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Process Control Block (PCB)
# ---------------------------------------------------------------------------
@dataclass
class PCB:
    pid: int
    name: str
    burst_time: int
    priority: int
    arrival_time: int = 0
    work_fn: callable = None
    data_chunk: list = None
    state: str = "NEW"
    remaining_time: int = field(init=False)
    start_time: float = None
    completion_time: float = None
    real_duration: float = None
    result: str = ""

    def __post_init__(self):
        self.remaining_time = self.burst_time


# ---------------------------------------------------------------------------
# 2. REAL work functions -- each does real analysis on real machine data
# ---------------------------------------------------------------------------
def work_failure_scan(pcb: PCB):
    """Counts real machine failures and breaks them down by failure type."""
    rows = pcb.data_chunk
    failures = [r for r in rows if r["Target"] == "1"]
    by_type = {}
    for r in failures:
        by_type[r["Failure Type"]] = by_type.get(r["Failure Type"], 0) + 1

    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_failure_scan.txt")
    with open(path, "w") as f:
        f.write(f"Machines scanned: {len(rows)}\n")
        f.write(f"Failures found: {len(failures)}\n")
        f.write("Breakdown by failure type:\n")
        for ftype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            f.write(f"  {ftype}: {count}\n")
    pcb.result = f"{len(failures)} failures / {len(rows)} machines scanned"


def work_temperature_stats(pcb: PCB):
    """Computes real air/process temperature statistics."""
    rows = pcb.data_chunk
    air = [float(r["Air temperature [K]"]) for r in rows]
    proc = [float(r["Process temperature [K]"]) for r in rows]

    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_temperature_stats.txt")
    with open(path, "w") as f:
        f.write(f"Air temp   -> min={min(air):.1f}K max={max(air):.1f}K avg={sum(air)/len(air):.2f}K\n")
        f.write(f"Process temp -> min={min(proc):.1f}K max={max(proc):.1f}K avg={sum(proc)/len(proc):.2f}K\n")
        f.write("Per-machine readings:\n")
        for r in rows:
            f.write(f"  {r['Product ID']}: air={r['Air temperature [K]']}K "
                    f"process={r['Process temperature [K]']}K\n")
    pcb.result = f"avg_air={sum(air)/len(air):.2f}K, avg_process={sum(proc)/len(proc):.2f}K"


def work_torque_analysis(pcb: PCB):
    """Analyzes real torque & rotational speed to flag overload risk."""
    rows = pcb.data_chunk
    torque = [float(r["Torque [Nm]"]) for r in rows]
    rpm = [float(r["Rotational speed [rpm]"]) for r in rows]
    high_torque_risk = [r for r in rows if float(r["Torque [Nm]"]) > 55]

    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_torque_analysis.txt")
    with open(path, "w") as f:
        f.write(f"Avg torque: {sum(torque)/len(torque):.2f} Nm\n")
        f.write(f"Avg rotational speed: {sum(rpm)/len(rpm):.1f} rpm\n")
        f.write(f"Machines with high torque (>55 Nm, overload risk): {len(high_torque_risk)}\n")
        for r in high_torque_risk:
            f.write(f"  {r['Product ID']}: torque={r['Torque [Nm]']}Nm rpm={r['Rotational speed [rpm]']}\n")
    pcb.result = f"avg_torque={sum(torque)/len(torque):.2f}Nm, high_risk_count={len(high_torque_risk)}"


def work_tool_wear_risk(pcb: PCB):
    """Scores real tool-wear data to flag machines needing maintenance soon."""
    rows = pcb.data_chunk
    wear = [int(r["Tool wear [min]"]) for r in rows]
    at_risk = [r for r in rows if int(r["Tool wear [min]"]) > 200]

    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_tool_wear_risk.txt")
    with open(path, "w") as f:
        f.write(f"Avg tool wear: {sum(wear)/len(wear):.1f} min\n")
        f.write(f"Max tool wear: {max(wear)} min\n")
        f.write(f"Machines needing maintenance soon (wear > 200min): {len(at_risk)}\n")
        for r in at_risk:
            f.write(f"  {r['Product ID']}: wear={r['Tool wear [min]']}min type={r['Type']}\n")
    pcb.result = f"avg_wear={sum(wear)/len(wear):.1f}min, at_risk_count={len(at_risk)}"


# ---------------------------------------------------------------------------
# 3. Process / Workload Manager
# ---------------------------------------------------------------------------
class ProcessManager:
    def __init__(self):
        self.processes = []
        self._next_pid = 1

    def submit_workload(self, name, burst_time, priority, work_fn, data_chunk):
        pcb = PCB(pid=self._next_pid, name=name, burst_time=burst_time,
                  priority=priority, work_fn=work_fn, data_chunk=data_chunk)
        pcb.state = "READY"
        self.processes.append(pcb)
        self._next_pid += 1
        print(f"[ProcessManager] Submitted {name} (PID={pcb.pid}, "
              f"est_burst={burst_time}, priority={priority}, rows={len(data_chunk)})")
        return pcb


# ---------------------------------------------------------------------------
# 4. Scheduling algorithm EVALUATION (on paper, using estimated burst times)
# ---------------------------------------------------------------------------
def evaluate_fcfs(processes):
    return _compute_stats(list(processes))


def evaluate_sjf(processes):
    return _compute_stats(sorted(processes, key=lambda p: p.burst_time))


def evaluate_priority(processes):
    return _compute_stats(sorted(processes, key=lambda p: p.priority))


def evaluate_round_robin(processes, quantum=3):
    queue = Queue()
    remaining = {p.pid: p.burst_time for p in processes}
    for p in processes:
        queue.put(p)
    clock = 0
    completion = {}
    while not queue.empty():
        p = queue.get()
        run = min(quantum, remaining[p.pid])
        clock += run
        remaining[p.pid] -= run
        if remaining[p.pid] > 0:
            queue.put(p)
        else:
            completion[p.pid] = clock
    waits, turns = [], []
    for p in processes:
        turnaround = completion[p.pid] - p.arrival_time
        waiting = turnaround - p.burst_time
        waits.append(waiting)
        turns.append(turnaround)
    return {
        "order": [p.name for p in processes],
        "avg_waiting": sum(waits) / len(waits),
        "avg_turnaround": sum(turns) / len(turns),
        "total_time": clock,
    }


def _compute_stats(order):
    clock = 0
    waits, turns, names = [], [], []
    for p in order:
        waiting = clock - p.arrival_time
        clock += p.burst_time
        turnaround = clock - p.arrival_time
        waits.append(waiting)
        turns.append(turnaround)
        names.append(p.name)
    return {
        "order": names,
        "avg_waiting": sum(waits) / len(waits),
        "avg_turnaround": sum(turns) / len(turns),
        "total_time": clock,
    }


def choose_best_algorithm(processes):
    candidates = {
        "FCFS": evaluate_fcfs(processes),
        "SJF": evaluate_sjf(processes),
        "Priority": evaluate_priority(processes),
        "Round Robin (q=3)": evaluate_round_robin(processes, quantum=3),
    }

    print("\n=== Scheduling Algorithm Comparison (estimated, before real execution) ===")
    print(f"{'Algorithm':<20}{'Order':<55}{'AvgWait':>10}{'AvgTurnaround':>15}{'Total':>8}")
    for algo, stats in candidates.items():
        print(f"{algo:<20}{', '.join(stats['order'])[:53]:<55}"
              f"{stats['avg_waiting']:>10.2f}{stats['avg_turnaround']:>15.2f}{stats['total_time']:>8}")

    best_algo = min(candidates, key=lambda a: candidates[a]["avg_waiting"])
    print(f"\n>>> Best algorithm chosen: {best_algo} "
          f"(lowest average waiting time = {candidates[best_algo]['avg_waiting']:.2f})")
    return best_algo, candidates[best_algo]


# ---------------------------------------------------------------------------
# 5. REAL execution using the chosen algorithm's order, with synchronization
# ---------------------------------------------------------------------------
resource_lock = threading.Lock()


def execute_process(pcb: PCB):
    pcb.state = "RUNNING"
    pcb.start_time = time.time()
    t0 = time.time()

    pcb.work_fn(pcb)  # real analysis on real data happens here

    pcb.real_duration = time.time() - t0
    pcb.completion_time = time.time()
    pcb.state = "TERMINATED"

    with resource_lock:
        print(f"[Executed] {pcb.name} (PID={pcb.pid}) -> {pcb.result} "
              f"| real_time={pcb.real_duration:.3f}s")


def run_with_best_order(processes, order_names):
    print("\n=== Real Execution (threads run concurrently, order reflects chosen algorithm) ===")
    name_to_pcb = {p.name: p for p in processes}
    ordered = [name_to_pcb[n] for n in order_names]

    threads = []
    start_clock = time.time()
    for pcb in ordered:
        t = threading.Thread(target=execute_process, args=(pcb,))
        threads.append(t)
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()
    return time.time() - start_clock


# ---------------------------------------------------------------------------
# 6. Final statistics report
# ---------------------------------------------------------------------------
def print_final_stats(processes, best_algo, best_stats, total_wall_time):
    print("\n=== Final Statistics ===")
    print(f"Chosen scheduling algorithm : {best_algo}")
    print(f"Predicted avg waiting time  : {best_stats['avg_waiting']:.2f} ticks")
    print(f"Predicted avg turnaround    : {best_stats['avg_turnaround']:.2f} ticks")
    print(f"Actual total wall-clock time: {total_wall_time:.3f} seconds\n")

    print(f"{'Workload':<24}{'PID':<5}{'Priority':<10}{'RealTime(s)':<13}{'Result (from real data)'}")
    for p in processes:
        print(f"{p.name:<24}{p.pid:<5}{p.priority:<10}{p.real_duration:<13.3f}{p.result}")

    print(f"\nOutput files written to ./{OUTPUT_DIR}/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  - {f} ({size:,} bytes)")


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Industrial AI Workload Management Platform — Demo (Real Dataset) ===\n")

    rows = load_real_dataset()
    chunks = split_into_chunks(rows, n=4)

    pm = ProcessManager()
    pm.submit_workload("Failure-Detection-AI", burst_time=2, priority=1,
                        work_fn=work_failure_scan, data_chunk=chunks[0])
    pm.submit_workload("Temperature-Monitor-AI", burst_time=5, priority=2,
                        work_fn=work_temperature_stats, data_chunk=chunks[1])
    pm.submit_workload("Torque-Analysis-AI", burst_time=3, priority=1,
                        work_fn=work_torque_analysis, data_chunk=chunks[2])
    pm.submit_workload("Tool-Wear-Risk-AI", burst_time=4, priority=3,
                        work_fn=work_tool_wear_risk, data_chunk=chunks[3])

    best_algo, best_stats = choose_best_algorithm(pm.processes)
    total_wall_time = run_with_best_order(pm.processes, best_stats["order"])
    print_final_stats(pm.processes, best_algo, best_stats, total_wall_time)

    print("\nDemo complete: real industrial machine sensor data (AI4I 2020 dataset) was")
    print("split across 4 AI workloads, scheduled using the best-performing algorithm,")
    print("and actually analyzed -- producing real failure/temperature/torque/wear reports.")
