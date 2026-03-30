import os
from typing import (
    Iterable,
    Optional,
    Sequence,
    Union,
)

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import pygplates
from gplately import (
    PlateReconstruction,
    EARTH_RADIUS,
)


_PathLike = Union[os.PathLike, str]
_FeatureCollectionInput = Union[
    pygplates.FeatureCollection,
    str,
    pygplates.Feature,
    Iterable[pygplates.Feature],
    Iterable[
        Union[
            pygplates.FeatureCollection,
            str,
            pygplates.Feature,
            Iterable[pygplates.Feature],
        ]
    ],
]
_RotationModelInput = Union[
    pygplates.RotationModel,
    _FeatureCollectionInput,
]


def run_calculate_convergence(

    min_time: float,
    max_time: float,
    temporal_resolution: int,
    rotation_model: Optional[Union[Sequence[str], str]] = None,
    topology_features: Optional[Sequence[str]] = None,
    static_polygons: Optional[Sequence[str]] = None,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    anchor_plate_id: int = 0,
    output_filename: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    
    use_parallel_func_from_files = False

    if plate_reconstruction is None:
        if n_jobs > 1:
            use_parallel_func_from_files = True
        else:
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
            )

    times = np.arange(min_time, max_time + temporal_resolution, temporal_resolution)

    if n_jobs == 1:
        data = [
            _tessellate_szs(
                plate_reconstruction=plate_reconstruction,
                time=t,
                ignore_warnings=True,
            )
            for t in times
        ]
    else:
        if not use_parallel_func_from_files:
            raise RuntimeError(
                "Parallel execution requires `topology_filenames` and `rotation_filenames`."
            )
        with Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0) as parallel:
            data = parallel(
                delayed(_tessellate_szs_parallel)(
                    rotation_model=rotation_model,
                    topology_features=topology_features,
                    static_polygons=static_polygons,
                    anchor_plate_id=anchor_plate_id,
                    time=t,
                    ignore_warnings=True,
                )
                for t in times
            )

    data = pd.concat(data)

    for col in (
        "distance_to_trench_edge (degrees)",
        "distance_from_trench_start (degrees)",
    ):
        if col in data.columns:
            x_km = np.deg2rad(data[col]) * EARTH_RADIUS
            data[col.replace("(degrees)", "(km)")] = x_km
            data = data.drop(columns=col, errors="ignore")
            
    # Save to CSV if a path is provided
    if output_filename:
        data.to_csv(output_filename, index=False)
        if verbose:
            print(f"Results written to: {output_filename}")

    return data


def _tessellate_szs(
    plate_reconstruction: PlateReconstruction,
    time: float,
    tessellation_threshold_radians: float = 0.001,
    ignore_warnings: bool = True,
) -> pd.DataFrame:
    
    data = plate_reconstruction.tessellate_subduction_zones(
        time=time,
        tessellation_threshold_radians=tessellation_threshold_radians,
        ignore_warnings=ignore_warnings,
        output_distance_to_nearest_edge_of_trench=True,
        output_distance_to_start_edge_of_trench=True,
        output_convergence_velocity_components=True,
        output_trench_absolute_velocity_components=True,
        output_subducting_absolute_velocity=True,
        output_subducting_absolute_velocity_components=True,
    )
    column_names = (
        "lon",
        "lat",
        "convergence_rate (cm/yr)",
        "convergence_obliquity (degrees)",
        "trench_velocity (cm/yr)",
        "trench_velocity_obliquity (degrees)",
        "arc_segment_length (degrees)",
        "trench_normal_angle (degrees)",
        "subducting_plate_ID",
        "trench_plate_ID",
        "distance_to_trench_edge (degrees)",
        "distance_from_trench_start (degrees)",
        "convergence_rate_orthogonal (cm/yr)",
        "convergence_rate_parallel (cm/yr)",
        "trench_velocity_orthogonal (cm/yr)",
        "trench_velocity_parallel (cm/yr)",
        "subducting_plate_absolute_velocity (cm/yr)",
        "subducting_plate_absolute_obliquity (degrees)",
        "subducting_plate_absolute_velocity_orthogonal (cm/yr)",
        "subducting_plate_absolute_velocity_parallel (cm/yr)",
    )
    out = pd.DataFrame(
        data,
        columns=column_names,
    )
    out["age (Ma)"] = np.float64(time)
    
    return out


def _tessellate_szs_parallel(
    rotation_model: Union[Sequence[str], str],
    topology_features: Sequence[str],
    static_polygons: Sequence[str],
    anchor_plate_id: int,
    time: float,
    tessellation_threshold_radians: float = 0.001,
    ignore_warnings: bool = True,
) -> pd.DataFrame:
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )
    
    return _tessellate_szs(
        plate_reconstruction=plate_reconstruction,
        time=time,
        tessellation_threshold_radians=tessellation_threshold_radians,
        ignore_warnings=ignore_warnings,
    )
