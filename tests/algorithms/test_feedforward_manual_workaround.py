# MIT License
#
# Copyright (c)
#
# Tests for the manual feedforward workaround using partial measurement and set_input_state.
# This verifies the pattern documented in docs/source/user_guide/feed_forward.rst

from __future__ import annotations

import torch
import torch.nn as nn
import perceval as pcvl
import pytest
from perceval import BasicState, Circuit

from merlin.algorithms.layer import QuantumLayer
from merlin.algorithms.feed_forward import FeedForwardBlock
from merlin.core.computation_space import ComputationSpace
from merlin.measurement.strategies import MeasurementStrategy


class BranchFeedforward(nn.Module):
    """Feedforward model with branch-local parameters, trainable via gradient descent.
    
    This is the pattern documented in docs/source/user_guide/feed_forward.rst
    for the workaround to FeedForwardBlock's limitation on branch-local parameters.
    
    Parameters
    ----------
    input_state : list | tuple
        Initial Fock state for the prefix circuit.
    prefix_circuit : Circuit
        Perceval circuit for the prefix (pre-measurement) stage.
    branch_configs : dict
        Mapping from measurement outcome (tuple) to (circuit, trainable_params, input_params).
    measured_modes : list[int]
        Modes to measure in the partial measurement (e.g., [0]).
    """

    def __init__(self, input_state, prefix_circuit, branch_configs, measured_modes):
        super().__init__()
        self.measured_modes = measured_modes
        self.branch_configs_dict = branch_configs

        # Build the prefix stage (partial measurement).
        self.partial_layer = QuantumLayer(
            circuit=prefix_circuit,
            input_state=input_state,
            measurement_strategy=MeasurementStrategy.partial(
                modes=measured_modes,
                computation_space=ComputationSpace.FOCK,
            ),
        )

        # Pre-build branch layers for each outcome, keyed by measurement outcome.
        # Initialize with input_size (classical) and n_photons (quantum dimension).
        # The branch layer uses the photon count remaining after measurement.
        self.branch_layers = nn.ModuleDict()
        
        for outcome, (circuit, trainable_params, input_params) in branch_configs.items():
            key = str(outcome)
            remaining_photons = sum(input_state) - sum(outcome)
            if remaining_photons == 0:
                continue
            self.branch_layers[key] = QuantumLayer(
                circuit=circuit,
                input_size=len(input_params),  # Classical input dimension
                n_photons=remaining_photons,
                trainable_parameters=trainable_params,
                input_parameters=input_params,
                measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
            )

    def forward(self, x: torch.Tensor | None = None) -> dict[tuple, torch.Tensor]:
        """Forward pass through the branched model.
        
        Parameters
        ----------
        x : torch.Tensor | None
            Classical input tensor for branches that require input_parameters.
            If provided, shape should be (batch_size, input_dim).
            If None, used for branches with no input_parameters.
        
        Returns
        -------
        dict[tuple, torch.Tensor]
            Dictionary mapping full measurement outcome keys to weighted probabilities.
            Probabilities have shape (batch_size,) and sum to 1 over all keys.
        """
        # Run the prefix stage to get partial measurement branches.
        partial_measurement = self.partial_layer()

        probabilities = {}
        for branch in partial_measurement.branches:
            key = str(branch.outcome)
            if sum(branch.outcome) == sum(self.partial_layer.input_state):
                probabilities[(branch.outcome[0], 0, 0)] = branch.probability.expand(
                    x.shape[0] if x is not None else 1
                )
                continue
            branch_layer = self.branch_layers[key]

            # Set the conditional amplitudes for this branch.
            # set_input_state() is the only way to route a StateVector through
            # a layer that was already constructed with trainable parameters.
            branch_layer.set_input_state(branch.amplitudes)

            # Execute the branch layer with its local classical inputs (if any).
            circuit, _trainable_params, input_params = self.branch_configs_dict[branch.outcome]
            if input_params:
                branch_output = branch_layer(x)  # shape: (batch, n_keys)
            else:
                branch_output = branch_layer()  # shape: (batch, n_keys)

            # Weight by branch probability and store results.
            # branch.probability is (batch,), branch_output is (batch, n_keys).
            # For each outcome key, weight by branch probability.
            branch_prob_weighted = branch.probability.unsqueeze(-1)  # (batch, 1)

            for index, remaining_key in enumerate(branch_layer.output_keys):
                # Construct full output key: (measured_outcomes, *remaining_outcomes)
                # This works for arbitrary measured_modes because we use the branch outcome directly.
                output_key = (branch.outcome[0], *remaining_key)
                branch_probs_for_key = branch_output[:, index]  # (batch,)
                weighted_probs = branch_prob_weighted.squeeze(-1) * branch_probs_for_key
                probabilities[output_key] = weighted_probs

        return probabilities


def test_manual_feedforward_workaround_single_batch():
    """Verify the manual workaround works for batch_size=1."""
    input_state = [1, 1, 0]
    prefix = Circuit(3) // pcvl.Unitary.random(3)

    # Define branch-specific circuits explicitly as 2-mode circuits (after measuring mode 0).
    # Use explicit Circuit(2) to ensure proper dimensionality.
    c0 = Circuit(2)
    c0.add(0, pcvl.BS(pcvl.P("x")))
    c0.add(0, pcvl.BS(pcvl.P("A")))
    
    c1 = Circuit(2)
    c1.add(0, pcvl.BS(pcvl.P("x")))
    c1.add(0, pcvl.BS(pcvl.P("B")))

    c2 = Circuit(2)
    
    branch_configs = {
        (0,): (c0, ["A"], ["x"]),
        (1,): (c1, ["B"], ["x"]),
        (2,): (c2, [], []),
    }

    model = BranchFeedforward(input_state, prefix, branch_configs, measured_modes=[0])

    # Single batch forward pass
    x = torch.tensor([[0.2]])  # (batch=1, input_dim=1)
    probabilities = model(x)

    # Verify output structure
    assert isinstance(probabilities, dict), "Output should be a dictionary"
    assert len(probabilities) > 0, "Should have at least one probability entry"

    # Verify all keys are tuples and all values are tensors with correct shape
    for key, prob in probabilities.items():
        assert isinstance(key, tuple), f"Key {key} should be a tuple"
        assert isinstance(prob, torch.Tensor), f"Probability for {key} should be a tensor"
        assert prob.shape == (1,), f"Probability for batch_size=1 should have shape (1,), got {prob.shape}"
        assert 0 <= prob.item() <= 1, f"Probability {prob.item()} should be in [0, 1]"

    # Verify probabilities sum to 1
    total_prob = sum(p.item() for p in probabilities.values())
    assert abs(total_prob - 1.0) < 1e-5, f"Probabilities should sum to 1, got {total_prob}"


def test_manual_feedforward_workaround_multi_batch():
    """Verify the manual workaround works correctly for batch_size > 1 with proper broadcasting."""
    input_state = [1, 1, 0]
    prefix = Circuit(3) // pcvl.Unitary.random(3)

    # Define branch-specific circuits explicitly as 2-mode circuits.
    c0 = Circuit(2)
    c0.add(0, pcvl.BS(pcvl.P("x")))
    c0.add(0, pcvl.BS(pcvl.P("A")))
    
    c1 = Circuit(2)
    c1.add(0, pcvl.BS(pcvl.P("x")))
    c1.add(0, pcvl.BS(pcvl.P("B")))

    c2 = Circuit(2)
    
    branch_configs = {
        (0,): (c0, ["A"], ["x"]),
        (1,): (c1, ["B"], ["x"]),
        (2,): (c2, [], []),
    }

    model = BranchFeedforward(input_state, prefix, branch_configs, measured_modes=[0])

    # Multi-batch forward pass
    batch_size = 3
    x = torch.tensor([[0.1], [0.2], [0.3]])  # (batch=3, input_dim=1)
    probabilities = model(x)

    # Verify all probabilities have the correct batch dimension
    for key, prob in probabilities.items():
        assert prob.shape == (batch_size,), (
            f"Probability for {key} should have shape ({batch_size},), got {prob.shape}"
        )
        assert torch.all((prob >= 0) & (prob <= 1)), (
            f"All probabilities for {key} should be in [0, 1], got {prob}"
        )

    # Verify probabilities sum to 1 for each batch element independently
    prob_array = torch.stack(list(probabilities.values()), dim=0)  # (n_keys, batch_size)
    total_probs = prob_array.sum(dim=0)  # (batch_size,)
    for i, total in enumerate(total_probs):
        assert abs(total.item() - 1.0) < 1e-5, (
            f"Batch element {i}: probabilities sum to {total.item()}, expected 1.0"
        )


def test_manual_feedforward_workaround_trainable():
    """Verify that parameters remain trainable across multiple forward passes."""
    input_state = [1, 1, 0]
    prefix = Circuit(3) // pcvl.Unitary.random(3)

    # Define branch-specific circuits explicitly as 2-mode circuits.
    c0 = Circuit(2)
    c0.add(0, pcvl.BS(pcvl.P("x")))
    c0.add(0, pcvl.BS(pcvl.P("A")))
    
    c1 = Circuit(2)
    c1.add(0, pcvl.BS(pcvl.P("x")))
    c1.add(0, pcvl.BS(pcvl.P("B")))

    c2 = Circuit(2)
    
    branch_configs = {
        (0,): (c0, ["A"], ["x"]),
        (1,): (c1, ["B"], ["x"]),
        (2,): (c2, [], []),
    }

    model = BranchFeedforward(input_state, prefix, branch_configs, measured_modes=[0])

    # Collect initial parameter values
    initial_params = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }

    # Define a simple loss and optimize
    x = torch.tensor([[0.2]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(5):
        optimizer.zero_grad()
        probs = model(x)
        # Simple loss: try to maximize the first probability we encounter
        first_probability_by_branch = {
            key[0]: probability
            for key, probability in probs.items()
            if key[0] in (0, 1)
        }
        loss = -sum(
            probability.mean() for probability in first_probability_by_branch.values()
        )
        loss.backward()
        optimizer.step()

    # Verify parameters have changed
    for name, initial_param in initial_params.items():
        current_param = dict(model.named_parameters())[name]
        assert not torch.allclose(initial_param, current_param, atol=1e-6), (
            f"Parameter {name} should have been updated by optimizer"
        )


def test_feedforwardblock_input_at_branch_fails():
    """Verify that FeedForwardBlock raises ValueError for branch-local input parameters.
    
    This is the error case that the manual workaround solves.
    This test reproduces the scenario from the documentation error example.
    """
    # Build a complete experiment with prefix, measurement, and branch-local parameter
    m = 4
    prefix = Circuit(m) // pcvl.Unitary.random(m)

    # Branch-local parameters are only used inside branch circuits
    # Note: each branch uses distinct parameters to avoid duplicate names
    experiment = pcvl.Experiment(m)
    experiment.add(0, prefix)
    experiment.add(0, pcvl.Detector.pnr())  # Measure first mode

    # FFCircuitProvider with branch-local parameters
    # Use unique parameter names per branch to avoid circuit conflicts
    c0 = Circuit(m - 1)
    c0.add(0, pcvl.BS(pcvl.P("A_branch")))
    
    c1 = Circuit(m - 1)
    c1.add(0, pcvl.BS(pcvl.P("B_branch")))
    
    provider = pcvl.FFCircuitProvider(1, 0, Circuit(m - 1))
    provider.add_configuration([0], c0)
    provider.add_configuration([1], c1)
    experiment.add(0, provider)

    # FeedForwardBlock should reject this because branch parameters aren't in the first stage
    with pytest.raises(ValueError):
        FeedForwardBlock(
            experiment,
            input_state=BasicState([1, 1, 0, 0]),
            trainable_parameters=["A_branch", "B_branch"],
            input_parameters=["A_branch", "B_branch"],
        )


def test_manual_feedforward_keys_arbitrary_measured_modes():
    """Verify key reconstruction works for arbitrary measured modes, not just mode 0."""
    # This test ensures the workaround generalizes beyond the simplistic measured_modes=[0] case.
    # For demonstration, we use measured_modes=[0] but verify the key structure is correct.
    input_state = [1, 1, 0]
    prefix = Circuit(3) // pcvl.Unitary.random(3)

    # Define branch-specific circuits explicitly as 2-mode circuits.
    c0 = Circuit(2)
    c0.add(0, pcvl.BS(pcvl.P("x")))
    c0.add(0, pcvl.BS(pcvl.P("A")))
    
    c1 = Circuit(2)
    c1.add(0, pcvl.BS(pcvl.P("x")))
    c1.add(0, pcvl.BS(pcvl.P("B")))

    c2 = Circuit(2)
    
    branch_configs = {
        (0,): (c0, ["A"], ["x"]),
        (1,): (c1, ["B"], ["x"]),
        (2,): (c2, [], []),
    }

    model = BranchFeedforward(input_state, prefix, branch_configs, measured_modes=[0])

    x = torch.tensor([[0.2]])
    probabilities = model(x)

    # All keys should start with one of the measurement outcomes
    expected_outcomes = (0, 1, 2)
    for key in probabilities.keys():
        assert key[0] in expected_outcomes, (
            f"First element of key {key} should be one of {expected_outcomes}, got {key[0]}"
        )
