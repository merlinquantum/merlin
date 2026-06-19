import os
import time
import csv
import torch
import psutil
from merlin.models.qcnn import QCNNClassifier 


log_dir = "./docs/source/quantum_expert_area/input_shape_logs/"
os.makedirs(log_dir, exist_ok=True)
csv_filename = os.path.join(log_dir, "scaling_study_benchmark.csv")


process = psutil.Process(os.getpid())

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

    print(f"Begining of the benchmark. Logs in {csv_filename}")
    
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
            

            memory_bytes = process.memory_info().rss
            peak_memory_mb = memory_bytes / (1024 * 1024)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run_time = time.perf_counter() - start_run
            
            writer.writerow([batch_size, img_h, num_params, peak_memory_mb, init_time, run_time, "SUCCESS"])
            print(f"OK | Params: {num_params} | RAM: {peak_memory_mb:.2f} MB | Init: {init_time:.4f}s | Run: {run_time:.4f}s")
            
        except Exception as e:
            writer.writerow([batch_size, img_h, "N/A", "N/A", "N/A", "N/A", f"FAILED: {type(e).__name__}"])
            print(f"ÉCHEC ({type(e).__name__})")
            #Here we stop the program to prevent it from crashing.
            if batch_size == phase2_batch:
                print("STOP")
                break 

print("Benchmark finished.")