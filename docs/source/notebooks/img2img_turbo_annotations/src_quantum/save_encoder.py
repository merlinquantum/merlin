from quantum_encoder import ParallelQuantumEncoder

pqencoder = ParallelQuantumEncoder((4, 16, 16), num_processes=5)

print("pqencoder defined")
df = {}
for p in pqencoder.bs.parameters:
    print(f"-{p.name}+{type(p)}")


# pqencoder_2 = ParallelQuantumEncoder((4, 16, 16), num_processes=5)
# pqencoder_2.bs.set_parameters(df)
"""df = {'quantum_params':pqencoder.bs.state_dict()}

print("-- pqencoder saved to state_dict")"""
