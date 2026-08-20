import os
import sys
import time
import csv
import torch
import subprocess

from merlin.models.qcnn import QCNNClassifier 

def get_current_ram_mb():
    """
    Take the current RAM consumption, regarding computer's os.
    """
    pid = os.getpid()
    
    # 1. LINUX : direct reading.
    if sys.platform == "linux":
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024.0
        except FileNotFoundError:
            pass

    # 2. MACOS : we use ps command.
    elif sys.platform == "darwin":
        try:
            # 'ps -o rss=' Returns the amount of memory in KB on a Mac
            rss_kb = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)])
            return int(rss_kb.strip()) / 1024.0
        except (subprocess.SubprocessError, ValueError):
            pass
            
    # 3. WINDOWS : We directly call the api via ctypes
    elif sys.platform == "win32":
        import ctypes
        
        # Structure to read memory on windows.
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
            
        process_handle = ctypes.windll.kernel32.GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        
        if ctypes.windll.psapi.GetProcessMemoryInfo(process_handle, ctypes.byref(counters), ctypes.sizeof(counters)):
            # WorkingSetSize is in bytes; divide it to get MB
            return counters.WorkingSetSize / (1024 * 1024)

    return 0.0 # Valeur de repli en cas d'erreur


log_dir = "./benchmarks"
os.makedirs(log_dir, exist_ok=True)
csv_filename = os.path.join(log_dir, "scaling_study_benchmark.csv")


num_classes = 2
phase1_height = 4
phase1_batches = list(range(1, 31))

phase2_batch = 1
phase2_heights = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

experiments = [(b, phase1_height) for b in phase1_batches] + \
              [(phase2_batch, h) for h in phase2_heights]


with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["batch_size", "image_height", "num_parameters", "peak_memory_mb", "init_time_sec", "forward_backward_time_sec", "status"])

    print(f"Beginning of the benchmark. Logs in {csv_filename}")
    
    for batch_size, img_h in experiments:
        input_size = (img_h, img_h)
        print(f"Test -> Batch: {batch_size}, Input: {input_size}...", end=" ", flush=True)
        
        try:
            start_init = time.perf_counter()
            qcnn = QCNNClassifier(input_size, num_classes)
            init_time = time.perf_counter() - start_init
            
            num_params = sum(p.numel() for p in qcnn.parameters() if p.requires_grad)

            x = torch.rand((batch_size, 1, input_size[0], input_size[1]))
            y = torch.randint(0, num_classes, (batch_size,))
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(qcnn.parameters(), lr=1e-3)

            _ = qcnn(x)

            start_run = time.perf_counter()
            
            logits = qcnn(x)
            loss = criterion(logits, y)
            

            peak_memory_mb = get_current_ram_mb()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run_time = time.perf_counter() - start_run
            
            writer.writerow([batch_size, img_h, num_params, peak_memory_mb, init_time, run_time, "SUCCESS"])
            print(f"OK | Params: {num_params} | RAM: {peak_memory_mb:.2f} MB | Init: {init_time:.4f}s | Run: {run_time:.4f}s")
            
        except Exception as e:
            writer.writerow([batch_size, img_h, "N/A", "N/A", "N/A", "N/A", f"FAILED: {type(e).__name__}"])
            print(f"ERROR ({type(e).__name__})")
            #Here we stop the program to prevent it from crashing.
            if batch_size == phase2_batch:
                print("STOP")
                break 

print("Benchmark finished.")