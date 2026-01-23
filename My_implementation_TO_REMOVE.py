import merlin as ML

quantum_layer = ML.FidelityKernel.simple(
    input_size=3, n_photons=3, trainable=True, input_state=[0, 0, 0, 0, 0]
)

print(quantum_layer)
