"""Quantum convolutional neural network model definitions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

import perceval as pcvl
import torch

from ..algorithms import QuantumLayer
from ..core import ComputationSpace, StateVector
from ..core.partial_measurement import PartialMeasurement
from ..measurement import MeasurementStrategy


class QCNNClassifier(torch.nn.Module):
    """Quantum convolutional neural network classifier.

    ``QCNNClassifier`` builds a trainable PyTorch model from a sequence of
    quantum convolution and quantum pooling stages followed by a quantum dense and
    a readout stage. The model is designed for square, single-channel image-like
    tensors. Each input image is amplitude-encoded into a two-photon
    :class:`~merlin.core.state_vector.StateVector`: one photon occupies a mode in
    the first register, the other occupies a mode in the second register, and
    each pixel value scales the basis state identified by its row and column.

    The resulting state vector is propagated through a
    :class:`torch.nn.Sequential` pipeline. ``QConv`` stages apply trainable
    Perceval beam-splitter circuits independently to each register. Within a
    given register, all convolution windows share the same trainable parameter;
    the two registers keep distinct trainable parameters. ``QPool`` stages
    partially measure selected modes, reinsert measured photons in the reduced
    register, continue the remaining QCNN pipeline on each valid measurement
    branch, and combine branch logits weighted by their probabilities. The final
    ``QDense`` stage applies a dense trainable photonic circuit, maps output
    probabilities to a tensor, and feeds them to a classical linear readout that
    returns class logits.

    Parameters
    ----------
    input_shape : tuple | list
        Spatial shape of the input image, written as ``(height, width)``. The
        current implementation requires a positive square shape.
    num_classes : int
        Number of output classes produced by the final readout layer.
    stages : list[QCNNClassifier.QConv | QCNNClassifier.QPool | QCNNClassifier.QDense] | None
        Optional stage specification. If omitted, the classifier uses
        ``QConv(kernel_size=2, stride=2)``, ``QPool(kernel_size=2)``, then
        ``QDense()``. Default is ``None``. An empty list is invalid and raises
        ``ValueError``.

    Attributes
    ----------
    num_classes : int
        Number of output classes.
    stages : list[QCNNClassifier.QConv | QCNNClassifier.QPool | QCNNClassifier.QDense] | None
        User-provided stage specification.
    layers : torch.nn.Sequential
        Executable PyTorch module containing the named quantum stages and
        readout. Individual layers can be inspected with integer indexing, for
        example ``classifier.layers[0]``.

    Raises
    ------
    TypeError
        If ``input_shape`` is not a tuple or list of integers, if
        ``num_classes`` is not an integer, or if ``stages`` is neither ``None``
        nor a list.
    ValueError
        If ``input_shape`` is not positive and square, if ``num_classes`` is not
        positive, or if ``stages`` does not satisfy the QCNN stage constraints.
    """

    _amplitude_basis_indices: torch.Tensor
    _input_shape: tuple[int, int]

    def __init__(
        self,
        input_shape: tuple[int, int] | list[int],
        num_classes: int,
        stages: (
            list[QCNNClassifier.QConv | QCNNClassifier.QPool | QCNNClassifier.QDense]
            | None
        ) = None,
    ):
        """Initialize and build the QCNN classifier.

        Parameters
        ----------
        input_shape : tuple | list
            Square image shape expected by :meth:`forward`.
        num_classes : int
            Number of output logits.
        stages : list[QCNNClassifier.QConv | QCNNClassifier.QPool | QCNNClassifier.QDense] | None
            Optional list made of
            :class:`~merlin.models.qcnn.QCNNClassifier.QConv`,
            :class:`~merlin.models.qcnn.QCNNClassifier.QPool`, and
            :class:`~merlin.models.qcnn.QCNNClassifier.QDense` stages.
            Default is ``None``.
        """
        super().__init__()
        self.num_classes = num_classes
        self.stages: list[QCNNClassifier._Stage] | None
        if stages is None:
            self.stages = None
        else:
            self.stages = list(stages)

        # Verify inputs
        if not isinstance(input_shape, (tuple, list)):
            raise TypeError("input_shape must have tuple or list type")
        if len(input_shape) != 2:
            raise ValueError("input_shape must be a tuple or list of size 2.")
        if not isinstance(input_shape[0], int) or not isinstance(input_shape[1], int):
            raise TypeError("input_shape elements must have int type")
        if type(input_shape[0]) is not int or type(input_shape[1]) is not int:
            raise ValueError("input_shape must contain integers.")
        validated_input_shape: tuple[int, int] = (input_shape[0], input_shape[1])
        self._input_shape = validated_input_shape
        if validated_input_shape[0] != validated_input_shape[1]:
            raise ValueError(
                "input_shape must represent a square (i.e. input_shape[0] == input_shape[1])."
            )
        if validated_input_shape[0] <= 0 or validated_input_shape[1] <= 0:
            raise ValueError("input_shape must contain values superior to 0.")

        if type(num_classes) is not int:
            raise TypeError("num_classes must have int type.")
        if num_classes <= 0:
            raise ValueError("num_classes must be superior to 0.")

        if stages is not None and type(stages) is not list:
            raise TypeError("stages must be None or have the list type.")

        self._resolved_stages = self.resolve_stages()
        self.layers = self.build_qcnn_model()
        self.register_buffer(
            "_amplitude_basis_indices",
            self._build_amplitude_basis_indices(),
            persistent=False,
        )

    @property
    def input_shape(self) -> tuple[int, int]:
        """Validated input image shape."""
        return copy.deepcopy(self._input_shape)

    @property
    def resolved_stages(self) -> list[QCNNClassifier._Stage]:
        """Validated stage sequence used to build the model."""
        return copy.deepcopy(self._resolved_stages)

    class _QCNNStageTypes(Enum):
        """Internal stage identifiers used in serialized configs."""

        QConv = "QConv"
        QPool = "QPool"
        QDense = "QDense"

    @dataclass
    class _Stage:
        """Base container for a QCNN stage.

        :meta public:

        Parameters
        ----------
        type : ``QCNNClassifier._QCNNStageTypes``
            Internal identifier of the stage kind.
        """

        def __init__(self, type: QCNNClassifier._QCNNStageTypes):
            self.type = type

    class QConv(_Stage):
        """Quantum convolution stage specification.

        A convolution stage applies the same construction rule to both quantum
        registers. Each sliding window is represented by a trainable
        beam-splitter mesh spanning ``kernel_size`` adjacent modes, and windows
        advance by ``stride`` modes. The convolution windows within one register
        share a single trainable beam-splitter parameter.

        Parameters
        ----------
        kernel_size : int
            Number of adjacent modes included in each quantum convolution
            window.
        stride : int
            Distance between consecutive convolution windows.

        Raises
        ------
        TypeError
            If ``kernel_size`` or ``stride`` is not an integer.
        ValueError
            If ``kernel_size`` or ``stride`` is not positive.
        """

        def __init__(self, kernel_size: int, stride: int):
            self.type = QCNNClassifier._QCNNStageTypes.QConv
            self.kernel_size = kernel_size
            self.stride = stride

            # Verification of inputs
            if type(kernel_size) is not int:
                raise TypeError("kernel_size must have int type.")
            if kernel_size <= 1:
                raise ValueError("kernel_size must be superior to 1.")
            if type(stride) is not int:
                raise TypeError("stride must have int type.")
            if stride <= 0:
                raise ValueError("stride must be superior to 0.")

        def __eq__(self, other):
            """Return whether two quantum convolution stages are identical."""
            if not isinstance(other, type(self)):
                return NotImplemented
            return (
                self.type == other.type
                and self.kernel_size == other.kernel_size
                and self.stride == other.stride
            )

        def __str__(self) -> str:
            """Return a concise string representation of the stage."""
            return f"QConv(kernel_size={self.kernel_size}, stride={self.stride})"

    class QPool(_Stage):
        """Quantum pooling stage specification.

        A pooling stage partially measures one mode per pooling window in each
        register. Valid measurement branches are propagated through the
        remaining stages and recombined by their probabilities.

        Parameters
        ----------
        kernel_size : int
            Number of adjacent modes covered by each pooling window.

        Raises
        ------
        TypeError
            If ``kernel_size`` is not an integer.
        ValueError
            If ``kernel_size`` is less than or equal to ``1``.
        """

        def __init__(self, kernel_size: int):
            self.type = QCNNClassifier._QCNNStageTypes.QPool
            self.kernel_size = kernel_size

            # Verification of input
            if type(kernel_size) is not int:
                raise TypeError("kernel_size must have int type.")
            if kernel_size <= 1:
                raise ValueError("kernel_size must be superior to 1.")

        def __eq__(self, other):
            """Return whether two quantum pooling stages are identical."""
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.type == other.type and self.kernel_size == other.kernel_size

        def __str__(self) -> str:
            """Return a concise string representation of the stage."""
            return f"QPool(kernel_size={self.kernel_size})"

    class QDense(_Stage):
        """Dense quantum readout stage specification.

        The dense stage is mandatory and must appear as the final QCNN stage. It
        applies a trainable beam-splitter circuit across all remaining modes
        before the classical readout layer.
        """

        def __init__(self):
            """Initialize a dense quantum readout stage."""
            self.type = QCNNClassifier._QCNNStageTypes.QDense

        def __eq__(self, other):
            """Return whether two dense quantum stages are identical."""
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.type == other.type

        def __str__(self) -> str:
            """Return a concise string representation of the stage."""
            return "QDense()"

    def resolve_stages(self) -> list[QCNNClassifier._Stage]:
        """Validate the stage specification and return the executable sequence.

        When no stage sequence is supplied, a default three-stage architecture is
        created. Custom stage sequences must end with exactly one dense stage,
        must use convolution kernels that fit the current register dimension,
        and must use pooling kernels that evenly reduce the current register
        dimension.

        Returns
        -------
        list[QCNNClassifier.QConv | QCNNClassifier.QPool | QCNNClassifier.QDense]
            Validated stage sequence used to build the QCNN.

        Raises
        ------
        ValueError
            If the stage order or stage parameters are incompatible with the
            current register dimensions, or if an unknown stage type is present.
        """
        # Check if stages is None
        if self.stages is None:
            resolved_stages: list[QCNNClassifier._Stage] = []
            # Default stages
            resolved_stages.append(QCNNClassifier.QConv(2, 2))
            resolved_stages.append(QCNNClassifier.QPool(2))
            resolved_stages.append(QCNNClassifier.QDense())
            return resolved_stages

        # Check if stages is empty
        if len(self.stages) == 0:
            raise ValueError(
                "Invalid stage specification: stages cannot be an empty list "
                "and it must include a QDense stage as its final stage."
            )

        # If stages were specified
        # Check that only last stage is QDense
        for i, stage in enumerate(self.stages):
            # Verify that stages are of the correct type
            if not isinstance(
                stage,
                (
                    QCNNClassifier.QConv,
                    QCNNClassifier.QPool,
                    QCNNClassifier.QDense,
                ),
            ):
                raise ValueError(
                    f"Invalid stage type: stage {i} has invalid type {type(stage)}; "
                    "stages must be of type QConv, QPool or QDense."
                )
            if stage.type == QCNNClassifier._QCNNStageTypes.QDense and i != (
                len(self.stages) - 1
            ):
                raise ValueError(
                    "Invalid stage specification: only last stage can be QDense"
                )
            if (
                i == (len(self.stages) - 1)
                and stage.type != QCNNClassifier._QCNNStageTypes.QDense
            ):
                raise ValueError(
                    "Invalid stage specification: last stage has to be QDense"
                )

        # Check QConv and QPool compatibility with current dimensions
        dim = self.input_shape[0]
        for stage in self.stages:
            if isinstance(stage, QCNNClassifier.QConv):
                # Appropriate conv if kernel_size <= dim of register
                appropriate_conv = dim >= stage.kernel_size
                if not appropriate_conv:
                    raise ValueError(
                        f"Invalid stage specification: current spatial dimension ({dim}) must be superior or equal to the convolution kernel size ({stage.kernel_size})."
                    )
                # Appropriate conv if kernel_size >= stride; else some modes are not covered
                appropriate_conv = stage.kernel_size >= stage.stride
                if not appropriate_conv:
                    raise ValueError(
                        f"Invalid stage specification: current convolution kernel size ({stage.kernel_size}) must be superior or equal to convolution stride ({stage.stride})."
                    )
                # Give freedom over the stride to the user. If the convolution window exceeds current dimension, it is not added to the circuit
                # TOVERIFY
            elif isinstance(stage, QCNNClassifier.QPool):
                appropriate_pooling = dim % stage.kernel_size == 0
                if not appropriate_pooling:
                    raise ValueError(
                        f"Invalid stage specification: current spatial dimension ({dim}) must be divisible by the pooling kernel size ({stage.kernel_size})."
                    )
                # Adjust current dimension after the pooling
                dim = dim - (dim // stage.kernel_size)

            elif isinstance(stage, QCNNClassifier.QDense):
                continue

            else:
                raise ValueError(
                    f"Invalid stage type; must be QConv, QPool or QDense but got: {type(stage)}"
                )

        return copy.deepcopy(self.stages)

    def summary(self):
        """Return a concise, human-readable description of the architecture.

        Returns
        -------
        str
            String containing input shape, number of classes, and resolved stage
            sequence.
        """
        stage_parts = []
        for stage in self._resolved_stages:
            if isinstance(stage, QCNNClassifier.QConv):
                stage_parts.append(
                    f"QConv(kernel_size={stage.kernel_size}, stride={stage.stride})"
                )
            elif isinstance(stage, QCNNClassifier.QPool):
                stage_parts.append(f"QPool(kernel_size={stage.kernel_size})")
            else:
                stage_parts.append("QDense()")

        stage_str = " -> ".join(stage_parts)
        return (
            "QCNNClassifier("
            f"input_shape={self.input_shape}, "
            f"num_classes={self.num_classes}, "
            f"stages=[{stage_str}]"
            ")"
        )

    def export_config(self):
        """Export a serializable architecture configuration.

        Weights are intentionally excluded. The returned dictionary can be used
        to reconstruct an equivalent ``QCNNClassifier`` architecture by
        rebuilding the stage objects from their serialized ``type`` entries.

        Returns
        -------
        dict
            Serializable architecture metadata containing ``input_shape``,
            ``num_classes``, and the resolved stage list under ``stages``.
        """
        serializable_stages = []
        for stage in self.resolved_stages:
            stage_cfg = {"type": stage.type.value}
            if isinstance(stage, QCNNClassifier.QConv):
                stage_cfg["kernel_size"] = stage.kernel_size
                stage_cfg["stride"] = stage.stride
            elif isinstance(stage, QCNNClassifier.QPool):
                stage_cfg["kernel_size"] = stage.kernel_size
            serializable_stages.append(stage_cfg)

        return {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "stages": serializable_stages,
        }

    @classmethod
    def from_config(cls, config: dict) -> QCNNClassifier:
        """Build a classifier from :meth:`export_config` metadata.

        Parameters
        ----------
        config : dict
            Serialized architecture metadata containing ``input_shape``,
            ``num_classes``, and ``stages``.

        Returns
        -------
        QCNNClassifier
            Reconstructed classifier with the same architecture.

        Raises
        ------
        ValueError
            If a serialized stage type is unknown.
        """
        stage_registry = {
            cls._QCNNStageTypes.QConv.value: cls.QConv,
            cls._QCNNStageTypes.QPool.value: cls.QPool,
            cls._QCNNStageTypes.QDense.value: cls.QDense,
        }

        stages = []
        for stage_config in config["stages"]:
            stage_type = stage_config["type"]
            try:
                stage_cls = stage_registry[stage_type]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid serialized QCNN stage type: {stage_type}"
                ) from exc

            kwargs = {
                key: value for key, value in stage_config.items() if key != "type"
            }
            stages.append(stage_cls(**kwargs))

        return cls(config["input_shape"], config["num_classes"], stages)

    def to(self, *args, **kwargs):
        """Move the classifier and quantum-layer runtime state to a device or dtype."""
        super().to(*args, **kwargs)
        for layer in self.layers:
            if isinstance(layer, QuantumLayer):
                layer.to(*args, **kwargs)
        return self

    def build_qcnn_model(self):
        """Build the executable quantum-classical QCNN pipeline.

        Each resolved stage is converted into a
        :class:`~merlin.algorithms.layer.QuantumLayer`. Quantum
        convolution layers return amplitudes so that later quantum stages can
        continue operating on state vectors. Quantum pooling layers return
        :class:`~merlin.core.partial_measurement.PartialMeasurement` objects for
        branch processing. The final dense quantum layer returns probabilities
        that are consumed by a classical :class:`torch.nn.Linear` readout.

        Returns
        -------
        torch.nn.Sequential
            Sequential module containing all quantum stages followed by the
            linear readout.

        Raises
        ------
        TypeError
            If an unknown stage type is encountered while building the model.
        """
        empty_circuit = pcvl.Circuit(sum(self.input_shape))
        current_shape = self.input_shape
        qcnn_layers = []
        layer_names = []
        num_qconv = 0
        num_qpool = 0

        # Keep the last stage (QDense) for after
        for stage in self.resolved_stages[:-1]:
            if isinstance(stage, QCNNClassifier.QConv):
                num_qconv += 1
                conv_circuit = self.build_conv_circuit(current_shape, stage)
                conv_layer = QuantumLayer(
                    circuit=conv_circuit,
                    n_photons=2,
                    trainable_parameters=["px"],
                    measurement_strategy=MeasurementStrategy.amplitudes(
                        computation_space=ComputationSpace.FOCK
                    ),
                )
                qcnn_layers.append(conv_layer)
                layer_names.append(f"QConv_{num_qconv}")

            elif isinstance(stage, QCNNClassifier.QPool):
                num_qpool += 1
                measured_modes, reinsert_modes, new_shape = self.resolve_pooling_modes(
                    current_shape, stage
                )
                empty_circuit = pcvl.Circuit(sum(current_shape))
                pool_layer = QuantumLayer(
                    circuit=empty_circuit,
                    n_photons=2,
                    measurement_strategy=MeasurementStrategy.partial(
                        measured_modes, computation_space=ComputationSpace.FOCK
                    ),
                )
                qcnn_layers.append(pool_layer)
                layer_names.append(f"QPool_{num_qpool}")
                # Change shape of circuit after pooling
                current_shape = new_shape

            else:
                raise TypeError(f"Unknown stage type encountered: {type(stage)}")

        # Apply QDense (mandatory last stage)
        dense_circuit = self.build_dense_circuit(current_shape)
        dense_layer = QuantumLayer(
            circuit=dense_circuit,
            n_photons=2,
            trainable_parameters=["px"],
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )

        # Readout layer
        readout = torch.nn.Linear(dense_layer.output_size, self.num_classes)

        qcnn_model = torch.nn.Sequential()
        for name, layer in zip(layer_names, qcnn_layers, strict=False):
            qcnn_model.add_module(name, layer)
        qcnn_model.add_module("QDense", dense_layer)
        qcnn_model.add_module("Readout", readout)

        return qcnn_model

    def build_conv_circuit(self, shape, stage):
        """Build a quantum convolution circuit for both registers.

        The two registers remain separate: no beam splitter is added between
        modes belonging to different registers. Each register receives the same
        sliding-window convolution pattern. All convolution windows in the first
        register share ``px_first_register`` as their beam-splitter parameter,
        and all convolution windows in the second register share
        ``px_second_register``.

        Parameters
        ----------
        shape : tuple[int, int]
            Current dimensions of the two quantum registers.
        stage : QCNNClassifier.QConv
            Convolution stage specification.

        Returns
        -------
        pcvl.Circuit
            Perceval circuit implementing the convolution stage.
        """
        circuit = pcvl.Circuit(sum(shape))

        # First register
        first_register_param = pcvl.P("px_first_register")
        i = 0
        while i < shape[0] and (i + stage.kernel_size <= shape[0]):
            circuit = self.build_single_conv(
                i, stage.kernel_size, circuit, first_register_param
            )
            i += stage.stride

        # Second register
        second_register_param = pcvl.P("px_second_register")
        i = shape[0]
        while (i < shape[0] * 2) and (i + stage.kernel_size <= shape[0] * 2):
            circuit = self.build_single_conv(
                i, stage.kernel_size, circuit, second_register_param
            )
            i += stage.stride

        return circuit

    def build_single_conv(self, i, kernel_size, circuit, parameter):
        """Add one trainable convolution window to a circuit.

        A single window is implemented as a down-and-up beam-splitter mesh so a
        photon can mix across the modes covered by ``kernel_size``. Every beam
        splitter inserted by this window uses the supplied ``parameter`` so the
        caller can share trainable parameters across several windows.

        Parameters
        ----------
        i : int
            First mode of the convolution window.
        kernel_size : int
            Number of modes covered by the window.
        circuit : pcvl.Circuit
            Circuit to mutate with the trainable beam splitters.
        parameter : ``pcvl.P``
            Shared Perceval parameter used as the ``theta`` value for every beam
            splitter in the convolution window.

        Returns
        -------
        pcvl.Circuit
            Circuit with the added convolution window.
        """
        # Slide the BS down on the circuit
        for j in range(i, i + kernel_size - 1):
            circuit.add(j, pcvl.BS(theta=parameter))
        # Slide the BS up on the circuit
        for k in range(j - 1, i - 1, -1):
            circuit.add(k, pcvl.BS(theta=parameter))
        return circuit

    def resolve_pooling_modes(self, shape, stage):
        """Resolve measured modes and output shape for a pooling stage.

        The first mode of every pooling window is measured. A photon detected in
        that mode is reinserted into the following mode after the measurement
        branch is reduced, which preserves the two-photon QCNN representation for
        downstream layers.

        Parameters
        ----------
        shape : tuple[int, int]
            Current dimensions of the two quantum registers.
        stage : QCNNClassifier.QPool
            Pooling stage specification.

        Returns
        -------
        tuple[list[int], list[int], tuple[int, int]]
            Measured modes, reinsertion modes, and the reduced register shape.

        Raises
        ------
        ValueError
            If pooling would produce registers with different dimensions.
        """
        measured_modes = []
        reinsert_modes = []

        # First register
        i = 0
        while i < shape[0] and (i + stage.kernel_size <= shape[0]):
            measured_modes.append(i)
            reinsert_modes.append(i + 1)
            i += stage.kernel_size
        num_mesured_modes_0 = len(measured_modes)
        new_shape_0 = shape[0] - num_mesured_modes_0

        # Second register
        i = shape[0]
        while (i < shape[0] * 2) and (i + stage.kernel_size <= shape[0] * 2):
            measured_modes.append(i)
            reinsert_modes.append(i + 1)
            i += stage.kernel_size
        new_shape_1 = shape[1] - len(measured_modes) + num_mesured_modes_0

        if new_shape_0 != new_shape_1:
            raise ValueError(
                "New shape after QPool must have the same shape[0] and shape[1] "
                f"but got new_shape[0]: {new_shape_0} and new_shape[1]: {new_shape_1}"
            )

        return measured_modes, reinsert_modes, (new_shape_0, new_shape_1)

    def build_dense_circuit(self, shape):
        """Build the dense quantum readout circuit.

        The circuit is a trainable beam-splitter mesh spanning all remaining
        modes, following the dense QCNN construction presented by Monbroussou et
        al.

        Parameters
        ----------
        shape : tuple[int, int]
            Current dimensions of the two quantum registers.

        Returns
        -------
        pcvl.Circuit
            Perceval circuit used by the final quantum dense layer.
        """
        circuit = pcvl.Circuit(sum(shape))

        # Slide the BS down the circuit
        i = 0
        while i < sum(shape) - 1:
            circuit.add(i, pcvl.BS(theta=pcvl.P(f"px_dense_bs_down_{i}")))
            # Add BS above if there is space
            j = i - 2
            while j > -1:
                circuit.add(j, pcvl.BS(theta=pcvl.P(f"px_dense_bs_down_{i}_{j}")))
                j -= 2
            i += 1

        # Slide the BS up the circuit
        i = sum(shape) - 3
        while i > -1:
            circuit.add(i, pcvl.BS(theta=pcvl.P(f"px_dense_bs_up_{i}")))
            # Add BS above if there is space
            j = i - 2
            while j > -1:
                circuit.add(j, pcvl.BS(theta=pcvl.P(f"px_dense_bs_up_{i}_{j}")))
                j -= 2
            i -= 1

        return circuit

    def forward(self, x):
        """Evaluate the classifier on a batch of images.

        The input tensor is amplitude-encoded, propagated through the resolved
        quantum stages, and finally mapped to class logits by the readout layer.
        If a pooling stage is encountered, its measurement branches are handled
        by :meth:`postprocess_pooling`, which continues the remaining stages on
        each valid branch and returns the probability-weighted logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, 1, input_shape[0],
            input_shape[1])``. The second dimension is the channel dimension;
            only one channel is currently supported.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch_size, num_classes)``.

        Raises
        ------
        TypeError
            If the model pipeline does not return a tensor.
        ValueError
            If ``x`` does not have the expected rank, channel count, or spatial
            dimensions, or if the output logits do not have the expected shape.
        """
        if not len(x.shape) == 4:
            raise ValueError(
                "Model input is expected to have the shape: [batch_size, 1, input_size[0], input_size[1]]"
            )
        if not x.shape[1] == 1:
            raise ValueError(
                "Model input must only have 1 channel at its second dimension."
            )
        if not x.shape[2] == x.shape[3] == self.input_shape[0]:
            raise ValueError(
                "Third and fourth input dimension must fit with the specified input_shape."
            )
        batch_size = x.shape[0]

        # Encode x using amplitude encoding
        x = self.amplitude_encode(x)

        for layer_index, (name, layer) in enumerate(self.layers.named_children()):
            x = layer(x)
            if name[:5] == "QPool":
                x = self.postprocess_pooling(x, layer_index + 1)
                break

        logits = x
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "QCNNClassifier.forward expected the model pipeline to return "
                f"a torch.Tensor, but got {type(logits)}."
            )
        if logits.shape != (batch_size, self.num_classes):
            raise ValueError(
                "QCNNClassifier.forward expected logits with shape "
                f"({batch_size}, {self.num_classes}), but got {tuple(logits.shape)}."
            )
        return logits

    def recursive_forward(self, x: StateVector, layer_index: int):
        """Continue the QCNN pipeline from a specific layer index.

        This helper is used after pooling, where each valid measurement branch
        contains a separate state vector. The branch state is propagated through
        the remaining layers, and nested pooling stages are post-processed in the
        same way as in :meth:`forward`.

        Parameters
        ----------
        x : merlin.core.state_vector.StateVector
            Branch state vector with shape ``(batch_size, basis_size)``.
        layer_index : int
            Index of the next layer to execute in ``self.layers``.

        Returns
        -------
        torch.Tensor | merlin.core.state_vector.StateVector | PartialMeasurement
            Result produced by the remaining pipeline. In normal classifier use
            this resolves to logits with shape ``(batch_size, num_classes)``.

        Raises
        ------
        TypeError
            If a pooling layer does not return a
            :class:`~merlin.core.partial_measurement.PartialMeasurement`, or if
            pooling post-processing does not return a tensor.
        """
        x_current = x
        for new_layer_index, (name, layer) in enumerate(
            list(self.layers.named_children())[layer_index:], start=layer_index
        ):
            x_current = layer(x_current)
            if name.startswith("QPool"):
                if not isinstance(x_current, PartialMeasurement):
                    raise TypeError(
                        "QCNN pooling layers must return a PartialMeasurement, "
                        f"but layer {name!r} returned {type(x_current)}."
                    )
                x_current = self.postprocess_pooling(x_current, new_layer_index + 1)
                if not isinstance(x_current, torch.Tensor):
                    raise TypeError(
                        "QCNN pooling post-processing must return a torch.Tensor, "
                        f"but got {type(x_current)}."
                    )
                break

        return x_current

    def _build_amplitude_basis_indices(self) -> torch.Tensor:
        """Return the Fock-basis index used by each image pixel."""
        height, width = self.input_shape
        state_vector = StateVector(torch.tensor([]), height + width, 2)
        basis_indices: list[int] = []

        for i in range(height):
            for j in range(width):
                basic_state = [0] * (height + width)
                basic_state[i] = 1
                basic_state[height + j] = 1
                basis_index = state_vector.index(basic_state)
                if basis_index is None:
                    raise ValueError(
                        "Could not map QCNN amplitude-encoding pixel to a Fock basis index."
                    )
                basis_indices.append(basis_index)

        return torch.tensor(basis_indices, dtype=torch.long)

    def amplitude_encode(self, x: torch.Tensor):
        """Encode image pixels as amplitudes of a two-photon state vector.

        Pixel ``x[:, :, i, j]`` scales the basis state with one photon in mode
        ``i`` of the first register and one photon in mode
        ``input_shape[0] + j`` of the second register. The resulting state vector
        is normalized before being passed to the quantum layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, 1, input_shape[0],
            input_shape[1])``.

        Returns
        -------
        merlin.core.state_vector.StateVector
            Normalized amplitude-encoded batch with shape
            ``(batch_size, basis_size)``. The basis size is determined by
            ``sum(input_shape)`` modes and two photons.
        """
        batch_size = x.shape[0]
        state_vector = StateVector(
            torch.tensor([], device=x.device), sum(self.input_shape), 2
        )
        basis_size = state_vector.basis_size

        state_tensor = torch.zeros(
            (batch_size, basis_size), dtype=torch.complex64, device=x.device
        )
        pixel_amplitudes = x[:, 0, :, :].reshape(batch_size, -1).to(torch.complex64)
        basis_indices = self._amplitude_basis_indices.to(x.device)
        expanded_indices = basis_indices.unsqueeze(0).expand(batch_size, -1)
        state_tensor = state_tensor.scatter(1, expanded_indices, pixel_amplitudes)

        return StateVector.from_tensor(
            state_tensor, n_modes=sum(self.input_shape), n_photons=2
        ).normalize()

    def postprocess_pooling(self, x: PartialMeasurement, layer_index: int):
        """Process pooling measurement branches and combine their logits.

        Each valid measurement branch is checked against the two-register QCNN
        constraints, measured photons are reinserted into the reduced state, and
        the branch is propagated through the remaining layers. Branch outputs are
        weighted by their measurement probabilities and summed.

        Parameters
        ----------
        x : PartialMeasurement
            Partial measurement returned by a quantum pooling layer.
        layer_index : int
            Index of the next layer to execute after the pooling layer.

        Returns
        -------
        torch.Tensor
            Probability-weighted logits with shape
            ``(batch_size, num_classes)``.

        Raises
        ------
        TypeError
            If a remaining pipeline branch does not return a tensor.
        ValueError
            If a branch state has an unexpected photon count, if branch logits
            or probabilities have incompatible shapes, or if a forbidden
            measurement outcome has non-zero probability.
        """
        batch_size = x.branches[0].probability.shape[0]
        branches = x.branches
        measured_modes = x.measured_modes
        x_combine = torch.zeros(
            batch_size,
            self.num_classes,
            dtype=x.branches[0].probability.dtype,
            device=x.branches[0].amplitudes.device,
        )
        for branch in branches:
            outcome = branch.outcome
            probabilities = branch.probability
            state_vector = branch.amplitudes

            # Skip branches with zero-amplitude states and ignore zero-amplitude
            # batch rows to prevent NaNs in downstream normalization.
            amplitude_norm_sq = state_vector.tensor.abs().pow(2).sum(dim=-1)
            active_rows = amplitude_norm_sq > 0
            if not torch.any(active_rows):
                continue

            possible = self.verify_outcome(list(outcome), probabilities)
            if possible:
                # Reinsert measured photons
                insertions = 0
                for index, outcome_elem in enumerate(outcome):
                    # Maximum of two insertions since there are two photons
                    if insertions == 2:
                        break
                    if outcome_elem > 0:
                        measured_mode = measured_modes[index]
                        reinsert_mode = measured_mode - index
                        # Reinsertion
                        state_vector = self.reinsert_photon(
                            state_vector.clone(), reinsert_mode
                        )
                        insertions += 1

                if state_vector.n_photons != 2:
                    raise ValueError(
                        "QCNN pooling post-processing expects branch states with "
                        f"2 photons after reinsertion, but got {state_vector.n_photons}."
                    )
                # Continue QCNN pipeline on that specific state_vector among the mixed states
                # recursive_forward is called iteratively. We might optimize to run in parallel.
                if bool(torch.all(active_rows)):
                    x_result = self.recursive_forward(state_vector, layer_index)
                else:
                    active_state_vector = StateVector(
                        tensor=state_vector.tensor[active_rows],
                        n_modes=state_vector.n_modes,
                        n_photons=state_vector.n_photons,
                        _normalized=state_vector.is_normalized,
                    )
                    active_result = self.recursive_forward(
                        active_state_vector, layer_index
                    )
                    x_result = torch.zeros(
                        batch_size,
                        self.num_classes,
                        dtype=active_result.dtype,
                        device=active_result.device,
                    )
                    x_result[active_rows] = active_result

                if not isinstance(x_result, torch.Tensor):
                    raise TypeError(
                        "QCNN pooling branch execution must return a torch.Tensor, "
                        f"but got {type(x_result)}."
                    )
                if x_result.shape != (batch_size, self.num_classes):
                    raise ValueError(
                        "QCNN pooling branch logits must have shape "
                        f"({batch_size}, {self.num_classes}), "
                        f"but got {tuple(x_result.shape)}."
                    )
                if probabilities.shape != (batch_size,):
                    raise ValueError(
                        "QCNN pooling branch probabilities must have shape "
                        f"({batch_size},), but got {tuple(probabilities.shape)}."
                    )
                # Combine results from all mixed state computations
                probabilities_weight = probabilities.unsqueeze(1)
                weighted = probabilities_weight * x_result
                x_combine = x_combine + weighted

        if not isinstance(x_combine, torch.Tensor):
            raise TypeError(
                "QCNN pooling post-processing expected to combine branch outputs "
                f"into a torch.Tensor, but got {type(x_combine)}."
            )
        return x_combine

    def _raise_if_forbidden_outcome_has_probability(self, outcome, probabilities):
        """Raise when a forbidden pooling outcome has non-zero probability.

        Parameters
        ----------
        outcome : Sequence[int]
            Forbidden measured occupation pattern.
        probabilities : torch.Tensor
            Branch probabilities for every batch item.

        Raises
        ------
        ValueError
            If ``probabilities`` contains non-zero values.
        """
        if not torch.allclose(
            probabilities, torch.zeros_like(probabilities), atol=1e-6
        ):
            raise ValueError(
                "Forbidden QCNN pooling outcome has non-zero probability. "
                f"Expected all probabilities to be 0, but got {probabilities} "
                f"for outcome: {outcome}."
            )

    def verify_outcome(self, outcome, probabilities):
        """Validate whether a pooling measurement outcome is physically usable.

        A valid QCNN pooling outcome measures at most one photon per register and
        only contains ``0`` or ``1`` occupation values. Forbidden outcomes are
        expected to have zero probability and raise a ``ValueError`` if they do
        not.

        Parameters
        ----------
        outcome : Sequence[int]
            Measured occupation pattern for the pooled modes.
        probabilities : torch.Tensor
            Branch probabilities for every batch item.

        Returns
        -------
        bool
            Whether the outcome can be propagated through the QCNN pipeline.

        Raises
        ------
        ValueError
            If ``outcome`` does not contain one measured pattern per register,
            or if a forbidden outcome has non-zero probability.
        """
        possible_outcome = True
        if len(outcome) % 2 != 0:
            raise ValueError(
                "QCNN pooling outcomes must contain one measured pattern per "
                f"register, but got an odd-length outcome: {outcome}."
            )
        half_index = int(len(outcome) / 2)
        first_register_outcome = outcome[:half_index]
        second_register_outcome = outcome[half_index:]

        photon_measured = False
        for outcome_elem in first_register_outcome:
            # Forbidden to measure more than 1 photon in QCNN setting
            if outcome_elem > 1:
                possible_outcome = False
                self._raise_if_forbidden_outcome_has_probability(outcome, probabilities)
                break
            # Forbidden to measure more than 1 photon per register in QCNN setting
            if outcome_elem == 1:
                if photon_measured:
                    possible_outcome = False
                    self._raise_if_forbidden_outcome_has_probability(
                        outcome, probabilities
                    )
                else:
                    photon_measured = True
                continue
            # Forbidden to have measurement result different from 0 or 1 in QCNN setting
            if outcome_elem != 0:
                possible_outcome = False
                self._raise_if_forbidden_outcome_has_probability(outcome, probabilities)
                break

        photon_measured = False
        for outcome_elem in second_register_outcome:
            # Forbidden to measure more than 1 photon in QCNN setting
            if outcome_elem > 1:
                possible_outcome = False
                self._raise_if_forbidden_outcome_has_probability(outcome, probabilities)
                break
            # Forbidden to measure more than 1 photon per register in QCNN setting
            if outcome_elem == 1:
                if photon_measured:
                    possible_outcome = False
                    self._raise_if_forbidden_outcome_has_probability(
                        outcome, probabilities
                    )
                else:
                    photon_measured = True
                continue
            # Forbidden to have measurement result different from 0 or 1 in QCNN setting
            if outcome_elem != 0:
                possible_outcome = False
                self._raise_if_forbidden_outcome_has_probability(outcome, probabilities)
                break

        return possible_outcome

    def reinsert_photon(self, state_vector: StateVector, reinsert_mode: int):
        """Insert one photon into a state-vector basis.

        The operation maps amplitudes from the current basis to the basis with
        one additional photon while preserving the PyTorch computation graph.
        Amplitudes whose source states already contain a photon in
        ``reinsert_mode`` are skipped because inserting there would violate the
        QCNN branch structure.

        Parameters
        ----------
        state_vector : merlin.core.state_vector.StateVector
            State vector before photon reinsertion.
        reinsert_mode : int
            Mode in which to reinsert the measured photon.

        Returns
        -------
        merlin.core.state_vector.StateVector
            New state vector with the same number of modes and one additional
            photon.
        """
        state_tensor = state_vector.tensor
        batch_size = state_tensor.shape[0]
        states = list(state_vector.basis.iter_states())

        new_dummy_state_vector = StateVector(
            torch.tensor([0]),
            n_modes=state_vector.n_modes,
            n_photons=state_vector.n_photons + 1,
        )
        new_state_basis_size = new_dummy_state_vector.basis_size
        new_state_tensor = torch.zeros(
            batch_size,
            new_state_basis_size,
            dtype=state_tensor.dtype,
            device=state_tensor.device,
        )

        # Setup mapping from old state indices (no photon at mode reinsert_mode) to new state index (photon reinserted)
        for i, state in enumerate(states):
            # i is the fock_to_index(state) returned index on state_vector
            if state[reinsert_mode] == 0:
                new_state = state[:reinsert_mode] + (1,) + state[reinsert_mode + 1 :]
                new_state_index = new_dummy_state_vector.basis.fock_to_index(new_state)
                new_state_tensor[:, new_state_index] = state_tensor[:, i]

        new_state_vector = StateVector(
            new_state_tensor,
            n_modes=state_vector.n_modes,
            n_photons=state_vector.n_photons + 1,
        )
        return new_state_vector
