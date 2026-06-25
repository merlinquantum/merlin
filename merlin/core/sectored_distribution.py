from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.combinadics import Combinadics
from .computation_space import ComputationSpace


@dataclass
class SectorResult:
    """One photon-number sector of a probability output. If keys are not given, the Combinadics.enumerate_states will be used."""

    tensor: torch.Tensor
    n_modes: int
    n_photons: int
    computation_space: ComputationSpace = ComputationSpace.FOCK
    keys: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        """Create the photon number to sector index map."""
        if self.keys is None:
            self.keys = tuple(
                Combinadics(
                    scheme="fock", n=self.n_photons, m=self.n_modes
                ).enumerate_states()
            )

    def to(self, *args, **kwargs) -> SectorResult:
        """Return a new state vector moved or cast via ``torch.Tensor.to``.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`torch.Tensor.to`.
        **kwargs
            Keyword arguments forwarded to :meth:`torch.Tensor.to`.

        Returns
        -------
        SectorResult
            Converted sector result.
        """
        new_tensor = self.tensor.to(*args, **kwargs)
        return SectorResult(
            new_tensor,
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            keys=self.keys,
        )

    def clone(self) -> SectorResult:
        """Return a cloned ``SectorResult`` with identical metadata and normalization flag.

        Returns
        -------
        SectorResult
            Cloned sector result.
        """
        return SectorResult(
            self.tensor.clone(),
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            keys=self.keys,
        )

    def detach(self) -> SectorResult:
        """Return a detached ``SectorResult`` sharing data without gradients.

        Returns
        -------
        SectorResult
            Detached sector result.
        """
        return SectorResult(
            self.tensor.detach(),
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            keys=self.keys,
        )

    def requires_grad_(self, requires_grad: bool = True) -> SectorResult:
        """Set ``requires_grad`` on the underlying tensor and return self.

        Parameters
        ----------
        requires_grad : bool
            Whether gradients should be tracked.

        Returns
        -------
        SectorResult
            The updated instance.
        """
        self.tensor.requires_grad_(requires_grad)
        return self


@dataclass
class SectoredDistribution:
    """Probability output spanning multiple photon-number sectors."""

    sectors: tuple[SectorResult, ...]

    def __post_init__(self) -> None:
        """Create the photon number to sector index map."""
        self._photon_map = {
            self.sectors[i].n_photons: i for i in range(len(self.sectors))
        }

    def get_sector(self, n_photons: int) -> SectorResult:
        """Return the SectorResult associated with n_photons."""
        if n_photons not in self._photon_map.keys():
            raise ValueError("No SectorResult with that number of photons")
        return self.sectors[self._photon_map[n_photons]]

    def total_probability(self) -> torch.Tensor:
        """Returns the total probability across the sectors."""
        total_prob = torch.zeros(
            (), dtype=self.sectors[0].tensor.dtype, device=self.sectors[0].tensor.device
        )
        for sector in self.sectors:
            total_prob = total_prob + sector.tensor.sum()
        return total_prob

    def to(self, *args, **kwargs) -> SectoredDistribution:
        """Return a new ``SectoredDistribution`` with SectorResults moved/cast via ``torch.Tensor.to``.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`torch.Tensor.to`.
        **kwargs
            Keyword arguments forwarded to :meth:`torch.Tensor.to`.

        Returns
        -------
        SectoredDistribution
            Converted sectored distribution.
        """
        new_sectors = []
        for sector in self.sectors:
            new_sectors.append(sector.to(*args, **kwargs))
        return SectoredDistribution(tuple(new_sectors))

    def clone(self) -> SectoredDistribution:
        """Return a cloned SectoredDistribution with metadata and logical performance preserved.

        Returns
        -------
        SectoredDistribution
            Cloned sectored distribution.
        """
        new_sectors = []
        for sector in self.sectors:
            new_sectors.append(sector.clone())
        return SectoredDistribution(tuple(new_sectors))

    def detach(self) -> SectoredDistribution:
        """Return a detached ``SectoredDistribution`` sharing data without gradients.

        Returns
        -------
        SectoredDistribution
            Detached sectored distribution.
        """
        new_sectors = []
        for sector in self.sectors:
            new_sectors.append(sector.detach())
        return SectoredDistribution(tuple(new_sectors))

    def requires_grad_(self, requires_grad: bool = True) -> SectoredDistribution:
        """Set ``requires_grad`` on underlying tensors and return self.

        Parameters
        ----------
        requires_grad : bool
            Whether gradients should be tracked.

        Returns
        -------
        SectoredDistribution
            The updated instance.
        """
        for sector in self.sectors:
            sector.requires_grad_(requires_grad)
        return self

    def to_tensor(
        self, return_keys: bool = False
    ) -> torch.Tensor | tuple[list[tuple[int, ...]], torch.Tensor]:
        """Convert the SectoredDistribution to a single concatenated tensor.

        Concatenates probability tensors from all sectors in photon-number order.
        For batched distributions, maintains all leading batch dimensions and
        concatenates along the state dimension.

        Parameters
        ----------
        return_keys : bool
            If True, also return the list of keys corresponding to each state
            in the output tensor. Default is False.

        Returns
        -------
        torch.Tensor
            If return_keys is False: concatenated probability tensor of shape
            (total_states,) for unbatched or
            (batch_shape..., total_states) for batched.
        tuple[list[tuple[int, ...]], torch.Tensor]
            If return_keys is True: tuple of (keys, tensor) where keys is a list
            of tuples representing Fock occupation numbers for each state.

        Examples
        --------
        >>> sector1 = SectorResult(
        ...     tensor=torch.tensor([0.5, 0.5]),
        ...     n_modes=2,
        ...     n_photons=1,
        ... )
        >>> sector2 = SectorResult(
        ...     tensor=torch.tensor([0.3, 0.3, 0.4]),
        ...     n_modes=2,
        ...     n_photons=2,
        ... )
        >>> dist = SectoredDistribution(sectors=(sector1, sector2))
        >>> tensor = dist.to_tensor()
        >>> tensor.shape
        torch.Size([5])
        >>> keys, tensor = dist.to_tensor(return_keys=True)
        >>> len(keys)
        5
        """
        is_batched = len(self.sectors[0].tensor.shape) > 1
        output_shape: int | tuple[int, ...]
        if not is_batched:
            output_shape = sum(sector.tensor.shape[0] for sector in self.sectors)
        else:
            batch_shape = self.sectors[0].tensor.shape[:-1]
            output_shape = (
                *batch_shape,
                sum(sector.tensor.shape[-1] for sector in self.sectors),
            )
        output_tensor = torch.zeros(
            output_shape,
            device=self.sectors[0].tensor.device,
            dtype=self.sectors[0].tensor.dtype,
        )
        output_index = 0
        if return_keys:
            output_keys: list[tuple[int, ...]] = []

        # Create the tensor
        for sector in self.sectors:
            if is_batched:
                size_of_sector = sector.tensor.shape[-1]
                output_tensor[..., output_index : output_index + size_of_sector] = (
                    sector.tensor
                )
            else:
                size_of_sector = sector.tensor.shape[0]
                output_tensor[output_index : output_index + size_of_sector] = (
                    sector.tensor
                )

            output_index += size_of_sector
            if return_keys:
                output_keys.extend(sector.keys)

        if return_keys:
            return output_keys, output_tensor

        return output_tensor


def clean_sectored_distribution(dist: SectoredDistribution) -> SectoredDistribution:
    """Reorganize a SectoredDistribution by actual photon counts from output keys.

    When photon loss occurs (e.g., in measurement or noisy simulations), output
    states may have different photon numbers than their original sector declaration.
    This function reorganizes the distribution by computing the actual photon count
    (sum of coordinates) for each state and grouping states together that have the
    same actual photon count. States with identical keys are combined, preserving
    their probability values.

    For batched distributions, the operation is performed independently for each
    batch item while maintaining the batch dimensions.

    Parameters
    ----------
    dist : SectoredDistribution
        The distribution to reorganize. May contain out-of-place keys where the
        sum of key coordinates differs from the sector's declared n_photons.

    Returns
    -------
    SectoredDistribution
        A new SectoredDistribution reorganized by actual photon counts. Each sector
        contains only states with photon count matching its n_photons. If states
        with the same key appear in different input sectors, their probabilities
        are combined in the output sector.

    Notes
    -----
    - Batch dimensions are preserved: unbatched input produces unbatched output,
      batched input produces batched output with the same leading dimensions.
    - Total probability is conserved across the reorganization.
    - Output sectors preserve the original sector discovery order. Initial
      photon numbers follow the input SectoredDistribution order, and newly
      discovered photon numbers are appended when first encountered.
    - Keys are regenerated from Combinadics based on the actual photon counts,
      ensuring consistency with the Fock space structure.

    Examples
    --------
    >>> # Sector with photon loss: 2-photon input with 0, 1, and 2 photon outputs
    >>> sector = SectorResult(
    ...     tensor=torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15]),
    ...     n_modes=2,
    ...     n_photons=2,
    ...     keys=((2, 0), (1, 1), (0, 2), (0, 1), (0, 0)),
    ... )
    >>> dist = SectoredDistribution(sectors=(sector,))
    >>> cleaned = clean_sectored_distribution(dist)
    >>> len(cleaned.sectors)
    3
    >>> cleaned.get_sector(0).tensor
    tensor([0.1500])
    >>> cleaned.get_sector(1).tensor
    tensor([0.1500])
    >>> cleaned.get_sector(2).tensor
    tensor([0.2000, 0.3000, 0.2000])
    """
    photon_numbers = list(dist._photon_map.keys())
    sector_shape = dist.sectors[0].tensor.shape

    # Combinadics per photon sector for faster indexing
    combinadics_per_sector = {
        i: Combinadics(scheme="fock", n=i, m=dist.sectors[0].n_modes)
        for i in photon_numbers
    }
    # Creating keys per photon sector
    keys_per_sector = {
        i: combinadics_per_sector[i].enumerate_states() for i in photon_numbers
    }

    # Creating new sectors to add the corresponding probs
    if len(sector_shape) == 1:
        is_batched = False

        # Tensor per sector
        sectors = {
            i: torch.zeros(
                combinadics_per_sector[i].compute_space_size(),
                device=dist.sectors[0].tensor.device,
                dtype=dist.sectors[0].tensor.dtype,
            )
            for i in photon_numbers
        }
    else:
        is_batched = True
        batch_shape = sector_shape[:-1]
        # Tensor per sector
        sectors = {
            i: torch.zeros(
                (*batch_shape, combinadics_per_sector[i].compute_space_size()),
                device=dist.sectors[0].tensor.device,
                dtype=dist.sectors[0].tensor.dtype,
            )
            for i in photon_numbers
        }

    for photon_number in photon_numbers.copy():
        sector_to_fix = dist.get_sector(photon_number)
        if is_batched:
            for col_index_in_previous_sector, key in enumerate(sector_to_fix.keys):
                # Checking if the photon sector exists, otherwise add it to the sectors and keys
                number_of_photons_in_state = sum(key)
                if number_of_photons_in_state not in photon_numbers:
                    combinadics_per_sector[number_of_photons_in_state] = Combinadics(
                        scheme="fock",
                        n=number_of_photons_in_state,
                        m=dist.sectors[0].n_modes,
                    )
                    keys_per_sector[number_of_photons_in_state] = (
                        combinadics_per_sector[
                            number_of_photons_in_state
                        ].enumerate_states()
                    )

                    sectors[number_of_photons_in_state] = torch.zeros(
                        (
                            *batch_shape,
                            combinadics_per_sector[
                                number_of_photons_in_state
                            ].compute_space_size(),
                        ),
                        device=dist.sectors[0].tensor.device,
                        dtype=dist.sectors[0].tensor.dtype,
                    )
                    photon_numbers.append(number_of_photons_in_state)

                # Adding the corresponding probs
                column_in_new_tensor = combinadics_per_sector[
                    number_of_photons_in_state
                ].index(key)

                sectors[number_of_photons_in_state][..., column_in_new_tensor] = (
                    sectors[number_of_photons_in_state][..., column_in_new_tensor]
                    + (sector_to_fix.tensor[..., col_index_in_previous_sector])
                )

        else:
            for key, value in zip(
                sector_to_fix.keys, sector_to_fix.tensor, strict=True
            ):
                # Checking if the photon sector exists, otherwise add it to the sectors and keys
                number_of_photons_in_state = sum(key)
                if number_of_photons_in_state not in photon_numbers:
                    combinadics_per_sector[number_of_photons_in_state] = Combinadics(
                        scheme="fock",
                        n=number_of_photons_in_state,
                        m=dist.sectors[0].n_modes,
                    )
                    keys_per_sector[number_of_photons_in_state] = (
                        combinadics_per_sector[
                            number_of_photons_in_state
                        ].enumerate_states()
                    )

                    sectors[number_of_photons_in_state] = torch.zeros(
                        combinadics_per_sector[
                            number_of_photons_in_state
                        ].compute_space_size(),
                        device=dist.sectors[0].tensor.device,
                        dtype=dist.sectors[0].tensor.dtype,
                    )
                    photon_numbers.append(number_of_photons_in_state)

                # Adding the corresponding probs
                index_in_new_tensor = combinadics_per_sector[
                    number_of_photons_in_state
                ].index(key)

                sectors[number_of_photons_in_state][index_in_new_tensor] = (
                    sectors[number_of_photons_in_state][index_in_new_tensor] + value
                )

    sectors_to_return = tuple(
        SectorResult(
            sectors[i],
            n_modes=dist.sectors[0].n_modes,
            n_photons=i,
            keys=tuple(keys_per_sector[i]),
        )
        for i in photon_numbers
    )

    return SectoredDistribution(sectors=sectors_to_return)
