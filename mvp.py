"""
Industrial AI Workload Management Platform — Minimal Demo
-----------------------------------------------------------
A very small proof-of-concept showing the core OS ideas from the project:
  - Each industrial AI workload is represented as a Process (with a PCB)
  - Workloads are scheduled on the CPU using FCFS and Round Robin
  - Workloads "execute" concurrently using threads, synchronized with a Lock
      (standing in for the shared resource / critical-section problem)

Run:  python workload_demo.py
"""

import threading
import time
import random
from dataclasses import dataclass, field
from queue import Queue


# ---------------------------------------------------------------------------
# 1. Process Control Block (PCB)
# ---------------------------------------------------------------------------
@dataclass
class PCB:
    pid: int
    name: str
    burst_time: int          # time (in "ticks") the workload needs on CPU
    priority: int
    state: str = "NEW"       # NEW -> READY -> RUNNING -> TERMINATED
    remaining_time: int = field(init=False)

    def __post_init__(self):
        self.remaining_time = self.burst_time


# ---------------------------------------------------------------------------
# 2. Workload / Process Manager
# ---------------------------------------------------------------------------
class ProcessManager:
    def __init__(self):
        self.processes = []
        self._next_pid = 1

    def submit_workload(self, name, burst_time, priority=1):
        pcb = PCB(pid=self._next_pid, name=name, burst_time=burst_time, priority=priority)
        pcb.state = "READY"
        self.processes.append(pcb)
        self._next_pid += 1
        print(f"[ProcessManager] Submitted {name} (PID={pcb.pid}, burst={burst_time})")
        return pcb


# ---------------------------------------------------------------------------
# 3. CPU Scheduler (FCFS + Round Robin)
# ---------------------------------------------------------------------------
class CPUScheduler:
    def __init__(self, processes):
        self.processes = processes

    def fcfs(self):
        print("\n--- FCFS Scheduling ---")
        clock = 0
        for p in self.processes:
            p.state = "RUNNING"
            print(f"t={clock:3d} | Running {p.name} (PID={p.pid}) for {p.burst_time} ticks")
            clock += p.burst_time
            p.state = "TERMINATED"
        print(f"Total time: {clock} ticks")

    def round_robin(self, quantum=2):
        print(f"\n--- Round Robin Scheduling (quantum={quantum}) ---")
        queue = Queue()
        for p in self.processes:
            p.remaining_time = p.burst_time  # reset for this run
            queue.put(p)

        clock = 0
        while not queue.empty():
            p = queue.get()
            run_time = min(quantum, p.remaining_time)
            p.state = "RUNNING"
            print(f"t={clock:3d} | Running {p.name} (PID={p.pid}) for {run_time} ticks "
                  f"(remaining before={p.remaining_time})")
            clock += run_time
            p.remaining_time -= run_time

            if p.remaining_time > 0:
                p.state = "READY"
                queue.put(p)
            else:
                p.state = "TERMINATED"
        print(f"Total time: {clock} ticks")


# ---------------------------------------------------------------------------
# 4. Thread-based concurrent execution + synchronization
# ---------------------------------------------------------------------------
shared_resource_lock = threading.Lock()
shared_resource_log = []


def execute_workload(pcb: PCB):
    """Simulates a workload actually running and touching a shared resource."""
    with shared_resource_lock:  # critical section
        pcb.state = "RUNNING"
        time.sleep(random.uniform(0.1, 0.3))  # simulate work
        shared_resource_log.append(pcb.name)
        pcb.state = "TERMINATED"
        print(f"[Thread] {pcb.name} (PID={pcb.pid}) finished safely "
              f"(shared resource updated by {len(shared_resource_log)} workloads so far)")


def run_concurrent_demo(processes):
    print("\n--- Concurrent Execution with Thread Synchronization ---")
    threads = [threading.Thread(target=execute_workload, args=(p,)) for p in processes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("All workloads completed. Shared resource log order:", shared_resource_log)


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Industrial AI Workload Management Platform — Demo ===\n")

    pm = ProcessManager()
    pm.submit_workload("Vision-Inspection-AI", burst_time=5, priority=2)
    pm.submit_workload("Predictive-Maintenance-AI", burst_time=3, priority=1)
    pm.submit_workload("Robotic-Arm-Control-AI", burst_time=4, priority=3)
    pm.submit_workload("Quality-Check-AI", burst_time=2, priority=1)

    scheduler = CPUScheduler(pm.processes)
    scheduler.fcfs()
    scheduler.round_robin(quantum=2)

    run_concurrent_demo(pm.processes)

    print("\nDemo complete. This is a minimal illustration of PCB, scheduling,")
    print("and synchronized concurrent execution — the core OS concepts behind the project.")
