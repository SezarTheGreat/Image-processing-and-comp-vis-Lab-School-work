# Non-Preemptive Shortest Job First Scheduling

processes = [
    ("P1", 6),
    ("P2", 8),
    ("P3", 7),
    ("P4", 3)
]

# Sort processes based on burst time
processes.sort(key=lambda x: x[1])

waiting_time = 0
total_wt = 0
total_tat = 0

print("Process\tBurst Time\tWaiting Time\tTurnaround Time")

for pid, bt in processes:
    tat = waiting_time + bt
    print(f"{pid}\t\t{bt}\t\t{waiting_time}\t\t{tat}")
    
    total_wt += waiting_time
    total_tat += tat
    waiting_time += bt

n = len(processes)
print("\nAverage Waiting Time =", total_wt / n)
print("Average Turnaround Time =", total_tat / n)

# Preemptive Shortest Job First (SRTF)

processes = [
    ("P1", 0, 8),
    ("P2", 1, 4),
    ("P3", 2, 2),
    ("P4", 3, 1)
]

n = len(processes)
remaining = {p[0]: p[2] for p in processes}
completion = {}
time = 0

while len(completion) < n:
    available = [p for p in processes if p[1] <= time and p[0] not in completion]

    if available:
        current = min(available, key=lambda x: remaining[x[0]])
        remaining[current[0]] -= 1
        if remaining[current[0]] == 0:
            completion[current[0]] = time + 1
    time += 1

print("Process\tArrival\tBurst\tCompletion\tWaiting\tTurnaround")

total_wt = total_tat = 0

for pid, at, bt in processes:
    tat = completion[pid] - at
    wt = tat - bt
    total_wt += wt
    total_tat += tat
    print(f"{pid}\t\t{at}\t{bt}\t{completion[pid]}\t\t{wt}\t{tat}")

print("\nAverage Waiting Time =", total_wt / n)
print("Average Turnaround Time =", total_tat / n)