from typing import Optional

import cryojax.simulator as cxs
from jaxtyping import Array, Float

from ...._custom_types import ParticleStackInfo
from ..._pose_search import (
    HierarchicalSO3GridSearch,
    HierarchicalSO3GridSearchKnownOffsets,
    HierarchicalSO3GridSearchStarCentered,
)


def _make_image_model_with_stack_poses(
    volume: cxs.AbstractVolumeRepresentation,
    relion_stack: ParticleStackInfo,
):
    """
    Create an image model using the poses from the Relion stack.
    This is a helper function to avoid passing poses explicitly.
    """
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    return cxs.make_image_model(
        volume,
        relion_stack["parameters"]["image_config"],
        relion_stack["parameters"]["pose"],
        relion_stack["parameters"]["transfer_theory"],
        normalizes_signal=True,
    )


def _make_image_model_from_gmm_estimate_poses(
    gmm_volume: cxs.GaussianMixtureVolume,
    observed_image: Float[Array, "image_height image_width"],
    relion_stack: ParticleStackInfo,
    pose_grid_search: HierarchicalSO3GridSearch,
    whitening_filter: Optional[Float[Array, "image_height image_width"]],
    whiten_calculated_images: bool,
    lowpass_filter: Optional[Float[Array, "image_height image_width"]] = None,
    highpass_filter: Optional[Float[Array, "image_height image_width"]] = None,
    use_star_shift_center: bool = False,
) -> cxs.AbstractImageModel:
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    if use_star_shift_center:
        optimal_pose = pose_grid_search(
            gmm_volume,
            observed_image,
            relion_stack["parameters"]["image_config"],
            relion_stack["parameters"]["transfer_theory"],
            whitening_filter,
            whiten_calculated_images,
            lowpass_filter,
            highpass_filter,
            center_offset=relion_stack["parameters"]["pose"].offset_in_angstroms,
        )
    else:
        optimal_pose = pose_grid_search(
            gmm_volume,
            observed_image,
            relion_stack["parameters"]["image_config"],
            relion_stack["parameters"]["transfer_theory"],
            whitening_filter,
            whiten_calculated_images,
            lowpass_filter,
            highpass_filter,
        )

    return cxs.make_image_model(
        gmm_volume,
        relion_stack["parameters"]["image_config"],
        optimal_pose,
        relion_stack["parameters"]["transfer_theory"],
        normalizes_signal=True,
    )


def _make_image_model_from_gmm_known_offsets_estimate_rotation(
    gmm_volume: cxs.GaussianMixtureVolume,
    observed_image: Float[Array, "image_height image_width"],
    relion_stack: ParticleStackInfo,
    pose_grid_search: HierarchicalSO3GridSearchKnownOffsets,
    whitening_filter: Optional[Float[Array, "image_height image_width"]],
    whiten_calculated_images: bool,
    lowpass_filter: Optional[Float[Array, "image_height image_width"]] = None,
    highpass_filter: Optional[Float[Array, "image_height image_width"]] = None,
) -> cxs.AbstractImageModel:
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    known_offset = relion_stack["parameters"]["pose"].offset_in_angstroms
    optimal_pose = pose_grid_search(
        gmm_volume,
        observed_image,
        relion_stack["parameters"]["image_config"],
        relion_stack["parameters"]["transfer_theory"],
        whitening_filter,
        whiten_calculated_images,
        lowpass_filter,
        highpass_filter,
    )
    pose_with_known_offsets = cxs.QuaternionPose(
        wxyz=optimal_pose.wxyz,
        offset_x_in_angstroms=known_offset[0],
        offset_y_in_angstroms=known_offset[1],
    )

    return cxs.make_image_model(
        gmm_volume,
        relion_stack["parameters"]["image_config"],
        pose_with_known_offsets,
        relion_stack["parameters"]["transfer_theory"],
        normalizes_signal=True,
    )


def make_image_model_from_gmm(
    structure: Float[Array, "n_atoms 3"],
    observed_image: Float[Array, "image_height image_width"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    pose_grid_search: Optional[
        HierarchicalSO3GridSearch | HierarchicalSO3GridSearchKnownOffsets
        | HierarchicalSO3GridSearchStarCentered
    ],
    whitening_filter: Optional[Float[Array, "image_height image_width"]],
    whiten_calculated_images: bool,
    lowpass_filter: Optional[Float[Array, "image_height image_width"]] = None,
    highpass_filter: Optional[Float[Array, "image_height image_width"]] = None,
) -> cxs.AbstractImageModel:
    volume = cxs.GaussianMixtureVolume(
        structure,
        amplitudes,
        variances,
    )
    if isinstance(pose_grid_search, HierarchicalSO3GridSearchKnownOffsets):
        return _make_image_model_from_gmm_known_offsets_estimate_rotation(
            volume,
            observed_image,
            relion_stack,
            pose_grid_search,
            whitening_filter,
            whiten_calculated_images,
            lowpass_filter,
            highpass_filter,
        )
    elif isinstance(pose_grid_search, HierarchicalSO3GridSearchStarCentered):
        return _make_image_model_from_gmm_estimate_poses(
            volume,
            observed_image,
            relion_stack,
            pose_grid_search,
            whitening_filter,
            whiten_calculated_images,
            lowpass_filter,
            highpass_filter,
            use_star_shift_center=True,
        )
    elif isinstance(pose_grid_search, HierarchicalSO3GridSearch):
        return _make_image_model_from_gmm_estimate_poses(
            volume,
            observed_image,
            relion_stack,
            pose_grid_search,
            whitening_filter,
            whiten_calculated_images,
            lowpass_filter,
            highpass_filter,
        )
    elif pose_grid_search is None:
        return _make_image_model_with_stack_poses(volume, relion_stack)
    else:
        raise TypeError(
            "pose_grid_search must be one of "
            "`HierarchicalSO3GridSearch`, "
            "`HierarchicalSO3GridSearchKnownOffsets`, or `None`."
        )
