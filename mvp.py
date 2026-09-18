"""
Industrial AI Workload Management Platform — Extended Demo
-------------------------------------------------------------
This version goes a step further than a toy simulation:

  1. Defines REAL workloads that do actual work (write text files,
     compute hashes/checksums, do number-crunching) instead of just
     sleeping.
  2. Evaluates several CPU scheduling algorithms (FCFS, SJF, Priority,
     Round Robin) on paper first -- computing Waiting Time / Turnaround
     Time / Completion Time for each -- to decide which one is "best"
     (lowest average waiting time).
  3. Actually EXECUTES the workloads for real, in the order chosen by
     the winning algorithm, using threads + a Lock for synchronization.
  4. Prints a final statistics table (scheduling stats + real execution
     time + output files produced).

Run:  python workload_demo.py
Output text files are written to ./workload_outputs/
"""

import os
import time
import random
import hashlib
import threading
from dataclasses import dataclass, field
from queue import Queue

OUTPUT_DIR = "workload_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Process Control Block (PCB)
# ---------------------------------------------------------------------------
@dataclass
class PCB:
    pid: int
    name: str
    burst_time: int              # estimated CPU ticks needed (used for scheduling)
    priority: int                # lower number = higher priority
    arrival_time: int = 0
    work_fn: callable = None     # the REAL function this workload will run
    state: str = "NEW"
    remaining_time: int = field(init=False)
    # stats filled in after scheduling / execution
    start_time: float = None
    completion_time: float = None
    real_duration: float = None
    result: str = ""

    def __post_init__(self):
        self.remaining_time = self.burst_time


# ---------------------------------------------------------------------------
# 2. REAL work functions each workload will actually perform
# ---------------------------------------------------------------------------
def work_generate_report(pcb: PCB):
    """Simulates an inspection-AI writing a report file."""
    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_report.txt")
    with open(path, "w") as f:
        for i in range(50000):
            f.write(f"[{pcb.name}] inspection log line {i}: status=OK\n")
    pcb.result = f"wrote {path}"


def work_compute_checksum(pcb: PCB):
    """Simulates a predictive-maintenance-AI hashing sensor data."""
    data = os.urandom(2_000_000)
    digest = hashlib.sha256(data).hexdigest()
    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_checksum.txt")
    with open(path, "w") as f:
        f.write(f"SHA256 of sensor batch: {digest}\n")
    pcb.result = f"checksum={digest[:12]}..."


def work_number_crunch(pcb: PCB):
    """Simulates a robotic-arm-control-AI doing trajectory calculations."""
    total = 0.0
    for i in range(3_000_000):
        total += (i ** 0.5) * 0.0001
    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_trajectory.txt")
    with open(path, "w") as f:
        f.write(f"Computed trajectory value: {total:.4f}\n")
    pcb.result = f"trajectory_value={total:.4f}"


def work_quality_scan(pcb: PCB):
    """Simulates a quality-check-AI scanning a batch of 'products'."""
    path = os.path.join(OUTPUT_DIR, f"{pcb.name}_scan.txt")
    defects = 0
    with open(path, "w") as f:
        for i in range(20000):
            ok = random.random() > 0.02
            if not ok:
                defects += 1
            f.write(f"product_{i}: {'PASS' if ok else 'DEFECT'}\n")
    pcb.result = f"defects_found={defects}"


# ---------------------------------------------------------------------------
# 3. Process / Workload Manager
# ---------------------------------------------------------------------------
class ProcessManager:
    def __init__(self):
        self.processes = []
        self._next_pid = 1

    def submit_workload(self, name, burst_time, priority, work_fn):
        pcb = PCB(pid=self._next_pid, name=name, burst_time=burst_time,
                  priority=priority, work_fn=work_fn)
        pcb.state = "READY"
        self.processes.append(pcb)
        self._next_pid += 1
        print(f"[ProcessManager] Submitted {name} (PID={pcb.pid}, "
              f"est_burst={burst_time}, priority={priority})")
        return pcb


# ---------------------------------------------------------------------------
# 4. Scheduling algorithm EVALUATION (on paper, using estimated burst times)
# ---------------------------------------------------------------------------
def evaluate_fcfs(processes):
    order = list(processes)  # arrival order
    return _compute_stats(order)


def evaluate_sjf(processes):
    order = sorted(processes, key=lambda p: p.burst_time)
    return _compute_stats(order)


def evaluate_priority(processes):
    order = sorted(processes, key=lambda p: p.priority)
    return _compute_stats(order)


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
    waits, turns = [], []
    names = []
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
resource_lock = threading.Lock()  # guards shared console output ("shared resource")


def execute_process(pcb: PCB):
    pcb.state = "RUNNING"
    pcb.start_time = time.time()
    t0 = time.time()

    pcb.work_fn(pcb)  # <-- the REAL work happens here

    t1 = time.time()
    pcb.real_duration = t1 - t0
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
        time.sleep(0.05)  # stagger start slightly, like a scheduler dispatching
    for t in threads:
        t.join()
    total_wall_time = time.time() - start_clock
    return total_wall_time


# ---------------------------------------------------------------------------
# 6. Final statistics report
# ---------------------------------------------------------------------------
def print_final_stats(processes, best_algo, best_stats, total_wall_time):
    print("\n=== Final Statistics ===")
    print(f"Chosen scheduling algorithm : {best_algo}")
    print(f"Predicted avg waiting time  : {best_stats['avg_waiting']:.2f} ticks")
    print(f"Predicted avg turnaround    : {best_stats['avg_turnaround']:.2f} ticks")
    print(f"Actual total wall-clock time: {total_wall_time:.3f} seconds\n")

    print(f"{'Workload':<28}{'PID':<5}{'Priority':<10}{'RealTime(s)':<13}{'Result'}")
    for p in processes:
        print(f"{p.name:<28}{p.pid:<5}{p.priority:<10}{p.real_duration:<13.3f}{p.result}")

    print(f"\nOutput files written to ./{OUTPUT_DIR}/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  - {f} ({size:,} bytes)")


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Industrial AI Workload Management Platform — Extended Demo ===\n")

    pm = ProcessManager()
    pm.submit_workload("Quality-Check-AI", burst_time=2, priority=1, work_fn=work_quality_scan)
    pm.submit_workload("Vision-Inspection-AI", burst_time=5, priority=2, work_fn=work_generate_report)
    pm.submit_workload("Predictive-Maintenance-AI", burst_time=3, priority=1, work_fn=work_compute_checksum)
    pm.submit_workload("Robotic-Arm-Control-AI", burst_time=4, priority=3, work_fn=work_number_crunch)

    best_algo, best_stats = choose_best_algorithm(pm.processes)
    total_wall_time = run_with_best_order(pm.processes, best_stats["order"])
    print_final_stats(pm.processes, best_algo, best_stats, total_wall_time)

    print("\nDemo complete: workloads were scheduled, compared across algorithms,")
    print("and then actually executed doing real work (files written, hashes computed).")
