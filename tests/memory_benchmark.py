# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import os
import time

import perceval as pcvl
import pynvml
import torch
from pynvml_utils import nvidia_smi
from torch.amp import GradScaler, autocast

from merlin import OutputMappingStrategy, QuantumLayer

parser = argparse.ArgumentParser(description="Test MerLin on your GPU !")
parser.add_argument(
    "--modes", type=int, default=8, help="Number of modes of the generic interferometer"
)
parser.add_argument(
    "--photons",
    type=int,
    default=4,
    help="Number of photons of the generic interferometer",
)
parser.add_argument("--bs", type=int, default=128, help="Batch size")
parser.add_argument(
    "--type", type=str, default=torch.float32, help="Type of the input data"
)
parser.add_argument("--hp", type=bool, default=False, help="Set Half Precision")


def benchmark_bs(MODES=8, PHOTONS=4, BS=32, TYPE=torch.float32, set_hp=False):
    """
    Benchmark memory usage for a GenericInterferometer running with MerLin support.

    This function measures GPU memory consumption and saves the benchmarking results
    as a dictionary in the ./results/ directory.
    It consists of training phases and reflectivity to match a target distribution.

    Parameters
    ----------
    MODES : int, optional
        Number of optical modes in the simulation (default: 8)

    PHOTONS : int, optional
        Number of photons to simulate (default: 4)

    BS : int, optional
        Batch size (default: 32)

    TYPE : torch.dtype, optional
        PyTorch data type for computations (default: torch.float32)
        Options: torch.float32, torch.float64, torch.complex64, torch.complex128

    set_hp : bool, optional
        Whether to use high precision mode (default: False)
    """

    print(
        f"\n Testing the GPU with a Batch size of {BS} for a circuit with {MODES} modes and {MODES // 2} photons"
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No support for CPU monitoring yet (coming soon ! ;) )")
        return None

    print(f"On device: {device}")

    if device == "cuda:0":
        torch.cuda.empty_cache()

    # create circuit: generic interferometer with trainable beam splitters and phase shifters
    circuit = pcvl.GenericInterferometer(
        MODES,
        lambda i: (
            pcvl.BS(theta=pcvl.P(f"theta_1_{i}"))
            .add(0, pcvl.PS(pcvl.P(f"phase_1_{i}")))
            .add(0, pcvl.BS(theta=pcvl.P(f"theta_2_{i}")))
            .add(0, pcvl.PS(pcvl.P(f"phase_2_{i}")))
        ),
    )
    nb_parameters = len(circuit.get_parameters())

    print(f"\n ----- \nWorking on {MODES} modes with {PHOTONS} photons \n -----")

    ####################################
    # Quantum Layer initialisation ###
    ####################################

    # Building the input state from the number of photons
    input_state = [0] * MODES
    for k in range(PHOTONS):
        input_state[2 * k] = 1
    print("\n Initializing the Quantum Layer")
    t_start_layer = time.time()
    print(
        f"\n Create circuit with input state = {input_state} (nb photons = {sum(input_state)}, nb parameters = {nb_parameters})"
    )
    input_size = len([
        p.name
        for p in circuit.get_parameters()
        if p.name.startswith("theta") or p.name.startswith("phase")
    ])
    q_model = QuantumLayer(
        input_size=input_size,
        output_size=None,
        circuit=circuit,
        input_state=input_state,
        trainable_parameters=[],
        input_parameters=["phase", "theta"],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        device=device,
        no_bunching=True,
    )
    print(f"Checking device of qlayer = {q_model.device}")
    t_end_layer = time.time() - t_start_layer

    ###################################
    # Computing the target values ###
    ###################################

    t_init = time.time()
    print("\n Computing the target distribution")
    # fix targets (phases + thetas -> unitary -> distribution)
    lengths = [
        len([p.name for p in circuit.get_parameters() if p.name.startswith("phase")]),
        len([p.name for p in circuit.get_parameters() if p.name.startswith("theta")]),
    ]
    target_phases = 2 * torch.pi * torch.rand((BS, lengths[0]))
    target_thetas = torch.rand((BS, lengths[1]))

    if set_hp:
        with autocast(device_type="cuda", dtype=torch.float32):
            target_distribution = q_model(
                target_phases.to(device), target_thetas.to(device)
            )
        scaler = GradScaler()
    else:
        target_distribution = q_model(
            target_phases.to(device), target_thetas.to(device)
        )

    print(
        f"-> Target distribution = ({target_distribution.shape}) with sum = {torch.sum(target_distribution)}"
    )

    print("\n ----------------------------------------------- \n")

    print("\n TRAINING IS STARTING \n")
    ##########################################
    # training loop to match the targets ###
    ##########################################

    # initialization of the parameters
    phases = torch.rand((BS, lengths[0]), requires_grad=True, device=device, dtype=TYPE)
    thetas = torch.rand((BS, lengths[1]), requires_grad=True, device=device, dtype=TYPE)
    # optimizer
    optimizer = torch.optim.Adam([phases, thetas], lr=0.001)
    # objective function
    criterion = torch.nn.MSELoss(reduction="sum")

    N_EPOCHS = 5
    history_used = []
    torch_history = []
    torch_reserved_history = []
    history_forward = []
    history_backward = []
    print(f"\n -- Init done in {time.time() - t_init} s\n")
    print(f"Before training the model is on device = {q_model.device}")
    t_exp_start = time.time()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        # running with Half Precision
        t_start_epoch = time.time()
        if set_hp:
            with autocast(device_type="cuda", dtype=torch.float32):
                # Forward pass: circuit unitary -> probability distribution
                probs = q_model(phases, thetas)
                loss = criterion(probs, target_distribution.to(device))
                t_end_forward = time.time()
            # update with Half Precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_backward = time.time()
        else:
            # no Half Precision update
            probs = q_model(phases, thetas)
            loss = criterion(probs, target_distribution.to(device))
            t_end_forward = time.time()
            loss.backward()
            optimizer.step()
            t_backward = time.time()

        # t_backward = time.time()
        history_backward.append(t_backward - t_end_forward)
        history_forward.append(t_end_forward - t_start_epoch)
        print(f"\n --> Iteration {epoch + 1}/{N_EPOCHS}: Loss = {loss.item():.4f}, ")

        # memory monitoring
        nvsmi = nvidia_smi.getInstance()
        query_result = nvsmi.DeviceQuery("memory.free, memory.total")
        total_used = 0
        for i, gpu_info in enumerate(query_result["gpu"]):
            total = gpu_info["fb_memory_usage"]["total"]
            free = gpu_info["fb_memory_usage"]["free"]
            used = total - free
            total_used += used
            print(f"GPU {i}: {used} MB used out of {total} MB")

        history_used.append(total_used)
        torch_history.append(torch.cuda.memory_allocated() / (1024 * 1024))
        torch_reserved_history.append(torch.cuda.memory_reserved() / (1024 * 1024))
        print(
            f"Memory allocated for PyTorch "
            f"\n - allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB"
            f"\n - reserved: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB"
        )
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(
            f"With pynvml"
            f"\n - total: {info.total / (1024 * 1024):.2f} MB"
            f"\n - free: {info.free / (1024 * 1024):.2f} MB"
            f"\n - used: {info.used / (1024 * 1024):.2f} MB"
        )

    # final results
    t_exp = time.time() - t_exp_start
    final_probs = q_model(phases, thetas)
    print(
        f"\n Found final distribution = {final_probs} with sum {torch.sum(final_probs)}"
    )
    print(f" -> target = {target_distribution}")
    avg_memory_needed = sum(history_used) / len(history_used)
    avg_torch = sum(torch_history) / len(torch_history)
    avg_reserved = sum(torch_reserved_history) / len(torch_reserved_history)
    forward_time = sum(history_forward) / len(history_forward)
    backward_time = sum(history_backward) / len(history_backward)

    dict_results = {
        "Batch size": BS,
        "mode": MODES,
        "nb photons": PHOTONS,
        "avg memory": avg_memory_needed,
        "torch memory": avg_torch,
        "reserved memory": avg_reserved,
        "type": str(type),
        "hp": set_hp,
        "t_layer": t_end_layer,
        "t forward": forward_time,
        "t backward": backward_time,
        "t_exp": t_exp,
    }
    print(f"\n Final results: {dict_results}")

    save_experiment_results(dict_results, filename="exp_MerlinRelease.json")

    print(f"\n --- \n Done for mode {MODES} with {PHOTONS} photons \n --- \n")


def save_experiment_results(results, filename="bunched_results.json"):
    """
    Append experiment results to a JSON file.

    Args:
        results (dict): Dictionary containing experiment results (with float values)
        filename (str): Path to the JSON file to store results
    """
    filename = os.path.join("./results", filename)
    # Check if file exists and load existing data
    if os.path.exists(filename):
        try:
            with open(filename) as file:
                all_results = json.load(file)
        except json.JSONDecodeError:
            # Handle case where file exists but is empty or corrupted
            all_results = []
    else:
        all_results = []

    # Append new results
    all_results.append(results)

    # Write updated data back to file
    with open(filename, "w") as file:
        json.dump(all_results, file, indent=4)

    return len(all_results)


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.photons <= args.modes // 2, (
        "You cannot have more photons than half the number of modes"
    )
    assert args.photons > 0, "You need at least 1 photon !"

    benchmark_bs(
        MODES=args.modes,
        PHOTONS=args.photons,
        BS=args.bs,
        TYPE=args.type,
        set_hp=args.hp,
    )
