from typing import Any, Optional, Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.jax_util import filter_bmap
from jaxtyping import Array, Float

from ...utils import apply_filter as _apply_filter
from .geometry import (
    getbestneighbors_base_SO3,
    getbestneighbors_next_SO3,
    grid_SO3,
)
from .shift import compute_correlation_at_optimal_offset


def _loss_for_grid_search_single(
    quat: Float[Array, "4"],
    volume: cxs.AbstractVolumeRepresentation,
    target_image: Float[Array, "H W"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    corr_mask: Optional[Float[Array, "H W"]],
    whitening_filter: Optional[Float[Array, "H W"]],
    whiten_calculated_images: bool,
    lowpass_filter: Optional[Float[Array, "H W"]],
    highpass_filter: Optional[Float[Array, "H W"]],
) -> Tuple[Float[Array, ""], Float[Array, " 2"]]:
    """
    Computes the correlation at the optimal shift using Plancherel's theorem and
    Fourier's convolution theorem for a given quaternion.

    The function outputs the negative correlation as this is a loss function.
    """
    pose = cxs.QuaternionPose(
        offset_x_in_angstroms=0.0, offset_y_in_angstroms=0.0, wxyz=quat
    )

    computed_image_no_shift = _apply_filter(
        cxs.make_image_model(
            volume,
            image_config,
            pose,
            transfer_theory,
            normalizes_signal=True,
        ).simulate(),
        whitening_filter if whiten_calculated_images else None,
    )
    computed_image_no_shift = _apply_filter(computed_image_no_shift, lowpass_filter)
    computed_image_no_shift = _apply_filter(computed_image_no_shift, highpass_filter)

    # get the optimal shift
    correlation, optimal_offset = compute_correlation_at_optimal_offset(
        target_image,
        computed_image_no_shift,
        image_config.get_coordinate_grid(physical=True),
        corr_mask,
    )

    return -correlation, optimal_offset


_loss_for_grid_search_vmapped = eqx.filter_vmap(
    _loss_for_grid_search_single,
    in_axes=(0, None, None, None, None, None, None, None, None, None),
)


def _loss_for_grid_search(
    quats: Float[Array, "N 4"],
    volume: cxs.AbstractVolumeRepresentation,
    target_image: Float[Array, "H W"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    corr_mask: Optional[Float[Array, "H W"]],
    whitening_filter: Optional[Float[Array, "H W"]],
    whiten_calculated_images: bool,
    lowpass_filter: Optional[Float[Array, "H W"]],
    highpass_filter: Optional[Float[Array, "H W"]],
    *,
    batch_size: int,
) -> Tuple[Float[Array, " N"], Float[Array, "N 2"]]:
    """
    Evaluates `_loss_for_grid_search_single` over `quats`, vmapping only
    `batch_size` quaternions at a time (via `filter_bmap`, looping with
    `jax.lax.scan`) instead of vmapping the whole grid at once. This keeps
    memory use roughly flat as the SO(3) grid grows, at the cost of running
    in sequential chunks.

    Note `filter_bmap`'s callback is handed a whole chunk of shape
    `(batch_size, 4)` at a time (not a single quaternion), so it must itself
    be vmapped over that leading axis -- hence `_loss_for_grid_search_vmapped`
    rather than calling `_loss_for_grid_search_single` directly.
    """
    return filter_bmap(
        lambda quat_chunk: _loss_for_grid_search_vmapped(
            quat_chunk,
            volume,
            target_image,
            image_config,
            transfer_theory,
            corr_mask,
            whitening_filter,
            whiten_calculated_images,
            lowpass_filter,
            highpass_filter,
        ),
        quats,
        batch_size=batch_size,
    )


class _HierSO3GridSearchCarry(eqx.Module):
    losses: Float[Array, " N"]
    quats: Float[Array, "N 4"]
    offsets: Float[Array, "N 2"]
    grid_indices: Any
    curr_resolution: int


class HierarchicalSO3GridSearch(eqx.Module):
    """
    Perform a hierarchical grid search over SO3 to find the optimal pose
    using a shift-invariant metric based on correlation. Also provides
    the optimal shift for the pose found.

    **Attributes:**
    - `base_quats`:
        Base quaternions for the SO3 grid at the coarsest resolution.
    - `n_rounds`:
        Number of rounds to perform the hierarchical search.
    - `n_candidates`:
        Number of candidate quaternions to consider in each round.
    - `base_grid_res`:
        Base grid resolution for the SO3 grid.
    - `corr_mask`:
        Mask for the correlation image to focus on specific regions during the search.
    - `grid_search_batch_size`:
        Number of candidate quaternions to evaluate at once (via `filter_bmap`)
        when scoring a round of the grid search. The base round evaluates
        `12 * Nside**2 * 6 * Nside` quaternions (e.g. 576 at `base_grid_res=1`),
        each requiring a full image simulation from `volume` -- lower this if
        that round runs out of memory, at the cost of running in more
        sequential chunks.
    - `max_shift_in_angstroms`:
        Largest in-plane offset, measured from the box centre, that the search may
        assign. Without it the offset is read off the whole cross-correlation
        image, so a neighbouring particle inside the box can win. `None` for no
        limit.
    """

    base_quats: Float[Array, "N 4"]
    n_rounds: int
    n_candidates: int
    base_grid_res: int
    corr_mask: Optional[Float[Array, "H W"]]
    grid_search_batch_size: int
    max_shift_in_angstroms: Optional[float]

    def __init__(
        self,
        base_grid_res,
        n_rounds,
        n_candidates,
        corr_mask=None,
        grid_search_batch_size=64,
        max_shift_in_angstroms=None,
    ):
        assert base_grid_res >= 1, "Base grid must be at least 1."
        assert n_rounds >= 0, "Number of rounds must be non-negative."
        assert n_candidates >= 1, "Number of candidates must be at least 1."
        assert grid_search_batch_size >= 1, "grid_search_batch_size must be at least 1."
        assert max_shift_in_angstroms is None or max_shift_in_angstroms > 0, (
            "max_shift_in_angstroms must be positive."
        )

        self.base_grid_res = base_grid_res
        self.base_quats = grid_SO3(base_grid_res)
        self.n_rounds = n_rounds
        self.n_candidates = n_candidates
        self.corr_mask = corr_mask
        self.grid_search_batch_size = grid_search_batch_size
        self.max_shift_in_angstroms = (
            None if max_shift_in_angstroms is None else float(max_shift_in_angstroms)
        )

    def _correlation_mask(
        self, image_config: cxs.BasicImageConfig, center_offset=None
    ) -> Optional[Float[Array, "H W"]]:
        """Restrict correlation shifts to a radius around center_offset (angstroms)."""
        if self.max_shift_in_angstroms is None:
            return self.corr_mask
        offsets = image_config.get_coordinate_grid(physical=True)
        if center_offset is None:
            center_offset = jnp.zeros((2,), dtype=offsets.dtype)
        else:
            center_offset = jnp.asarray(center_offset, dtype=offsets.dtype)
        within = (
            jnp.linalg.norm(offsets - center_offset, axis=-1)
            <= self.max_shift_in_angstroms
        ).astype(offsets.dtype)
        return within if self.corr_mask is None else self.corr_mask * within

    def __call__(
        self,
        volume: cxs.AbstractVolumeRepresentation,
        image: Float[Array, "H W"],
        image_config: cxs.BasicImageConfig,
        transfer_theory: cxs.ContrastTransferTheory,
        whitening_filter: Optional[Float[Array, "H W"]],
        whiten_calculated_images: bool,
        lowpass_filter: Optional[Float[Array, "H W"]] = None,
        highpass_filter: Optional[Float[Array, "H W"]] = None,
        center_offset=None,
    ) -> cxs.QuaternionPose:
        """Perform the hierarchical SO3 grid search to find the optimal pose.

        **Arguments:**
        - `volume`:
            `cryojax` `AbstractVolumeRepresentation` representing a volumetric density.
        - `image`:
            Target image to compare against through the search.
        - `image_config`:
            Configuration for image simulation.
        - `transfer_theory`:
            Contrast transfer theory to use for image simulation.
        - `whitening_filter`:
            Whitening filter applied to computed images during the search.
        - `whiten_calculated_images`:
            Whether to apply the whitening filter to computed images.
        - `lowpass_filter`:
            Optional lowpass filter applied to computed images (frequency marching).
        - `highpass_filter`:
            Optional highpass filter applied to computed images.

        **Returns:**
        - `optimal_pose`:
            The optimal pose found through the hierarchical SO3 grid search.
        """

        image = jnp.asarray(image)
        corr_mask = self._correlation_mask(image_config, center_offset=center_offset)
        losses, offsets = _loss_for_grid_search(
            self.base_quats,
            volume,
            image,
            image_config,
            transfer_theory,
            corr_mask,
            whitening_filter,
            whiten_calculated_images,
            lowpass_filter,
            highpass_filter,
            batch_size=self.grid_search_batch_size,
        )

        if self.n_rounds == 0:
            best_index = jnp.argmin(losses)
            optimal_pose = cxs.QuaternionPose(
                wxyz=self.base_quats[best_index],
                offset_x_in_angstroms=offsets[best_index, 0],
                offset_y_in_angstroms=offsets[best_index, 1],
            )

        else:
            # Do the first iteration outside the loop
            quats, grid_indices = getbestneighbors_base_SO3(
                losses,
                self.base_quats,
                N=self.n_candidates,
                base_resol=self.base_grid_res,
            )
            losses, offsets = _loss_for_grid_search(
                quats,
                volume,
                image,
                image_config,
                transfer_theory,
                corr_mask,
                whitening_filter,
                whiten_calculated_images,
                lowpass_filter,
                highpass_filter,
                batch_size=self.grid_search_batch_size,
            )
            carry = _HierSO3GridSearchCarry(
                losses=losses,
                quats=quats,
                grid_indices=grid_indices,
                offsets=offsets,
                curr_resolution=self.base_grid_res + 1,
            )
            # Then do the rest of the rounds in a loop
            carry = jax.lax.fori_loop(
                lower=1,
                upper=self.n_rounds,
                body_fun=lambda _, x: _run_global_SO3_step(
                    x,
                    self.n_candidates,
                    volume,
                    image,
                    image_config,
                    transfer_theory,
                    corr_mask,
                    whitening_filter,
                    whiten_calculated_images,
                    lowpass_filter,
                    highpass_filter,
                    batch_size=self.grid_search_batch_size,
                ),
                init_val=carry,
            )
            best_index = jnp.argmin(carry.losses)
            optimal_pose = cxs.QuaternionPose(
                wxyz=carry.quats[best_index],
                offset_x_in_angstroms=carry.offsets[best_index, 0],
                offset_y_in_angstroms=carry.offsets[best_index, 1],
            )

        return optimal_pose


class HierarchicalSO3GridSearchStarCentered(HierarchicalSO3GridSearch):
    """Search rotations and translations constrained near each particle's STAR offset."""


class HierarchicalSO3GridSearchKnownOffsets(HierarchicalSO3GridSearch):
    """
    Marker class for running the standard hierarchical SO3 search while using
    known offsets when constructing the final image model.
    """


def _run_global_SO3_step(
    hier_grid_search_carry: _HierSO3GridSearchCarry,
    n_candidates: int,
    volume: cxs.AbstractVolumeRepresentation,
    image: Float[Array, "H W"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    corr_mask: Optional[Float[Array, "H W"]],
    whitening_filter: Optional[Float[Array, "H W"]],
    whiten_calculated_images: bool,
    lowpass_filter: Optional[Float[Array, "H W"]] = None,
    highpass_filter: Optional[Float[Array, "H W"]] = None,
    *,
    batch_size: int,
):
    quats, grid_indices = getbestneighbors_next_SO3(
        hier_grid_search_carry.losses,
        hier_grid_search_carry.quats,
        hier_grid_search_carry.grid_indices,
        curr_res=hier_grid_search_carry.curr_resolution,
        N=n_candidates,
    )
    losses, offsets = _loss_for_grid_search(
        quats,
        volume,
        image,
        image_config,
        transfer_theory,
        corr_mask,
        whitening_filter,
        whiten_calculated_images,
        lowpass_filter,
        highpass_filter,
        batch_size=batch_size,
    )
    return _HierSO3GridSearchCarry(
        losses=losses,
        quats=quats,
        offsets=offsets,
        grid_indices=grid_indices,
        curr_resolution=hier_grid_search_carry.curr_resolution + 1,
    )


def local_SO3_hier_search(lossfn, base_grid_res, n_rounds, n_candidates):
    raise NotImplementedError(
        "Local SO3 hierarchical search is not implemented yet. "
        "Please use global SO3 hierarchical search instead."
    )
