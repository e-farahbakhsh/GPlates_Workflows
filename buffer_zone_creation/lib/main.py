import os
from sys import stderr
from typing import (
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)
import warnings

import geopandas as gpd
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import linemerge

import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    EARTH_RADIUS,
)
from gplately.geometry import (
    pygplates_to_shapely,
    wrap_geometries,
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


def run_create_buffer_zones(
    times: Sequence[float],
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = 0,
    clip_to_overriding_plate: bool = False,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = 6,
    n_jobs: int = -2,
    verbose: bool = False,
    return_output: bool = False,
) -> Optional[List[gpd.GeoDataFrame]]:

    if plate_reconstruction is None:
        if topology_features is None or rotation_model is None:
            raise TypeError(
                "Either `plate_reconstruction` or both "
                "`topology_features` and `rotation_model` "
                "must not be None."
            )

    if output_dir is not None and not os.path.isdir(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + output_dir,
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    times_split = np.array_split(times, n_jobs)
    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_multiple_timesteps_buffer)(
                times=t,
                buffer_distance=buffer_distance,
                return_output=return_output,
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                clip_to_overriding_plate=clip_to_overriding_plate,
                output_dir=output_dir,
            )
            for t in times_split
        )
        
    if return_output:
        out = []
        for i in results:
            out.extend(i)
        return out
    
    return None


def _multiple_timesteps_buffer(
    times: Sequence[float],
    buffer_distance: float,
    return_output: bool,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = 0,
    clip_to_overriding_plate: bool = False,
    output_dir: _PathLike = os.curdir,
):
    
    if plate_reconstruction is None:
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)
        if not isinstance(topology_features, pygplates.FeatureCollection):
            topology_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(topology_features).get_features()
                )
        if not isinstance(static_polygons, pygplates.FeatureCollection):
            static_polygons = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(static_polygons).get_features()
                )

    out = []
    for time in times:
        out.append(
            _create_buffer_zones(
                time=time,
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                clip_to_overriding_plate=clip_to_overriding_plate,
                output_dir=output_dir,
                buffer_distance=buffer_distance,
                return_output=return_output,
            )
        )
        
    if return_output:
        return out


def _create_buffer_zones(
    time: float,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = 0,
    clip_to_overriding_plate: bool = False,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = 6,
    return_output: bool = False,
) -> Optional[gpd.GeoDataFrame]:

    if plate_reconstruction is None:
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)
        if not isinstance(topology_features, pygplates.FeatureCollection):
            topology_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(topology_features).get_features()
                )
        if not isinstance(static_polygons, pygplates.FeatureCollection):
            static_polygons = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(static_polygons).get_features()
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
            )
    else:
        rotation_model = plate_reconstruction.rotation_model
        topology_features = plate_reconstruction.topology_features
        static_polygons = plate_reconstruction.static_polygons

    gplot = PlotTopologies(plate_reconstruction, anchor_plate_id=anchor_plate_id)
    gplot.time = float(time)
    plate_polygons = gplot.get_all_topologies()
    plate_polygons["feature_type"] = plate_polygons["feature_type"].astype(str)
    plate_polygons = plate_polygons[
        plate_polygons["feature_type"].isin({
            "gpml:TopologicalClosedPlateBoundary",
            "gpml:OceanicCrust",
            "gpml:TopologicalNetwork",
        })
    ]

    topologies = _extract_overriding_plates(
        time=time,
        topology_features=topology_features,
        rotation_model=rotation_model,
        anchor_plate_id=anchor_plate_id,
    )
    plate_polygons.crs = topologies.crs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        topologies = topologies[
            (topologies["over"] != -1)
            & (topologies["over"] != 0)
            & (topologies["polarity"] != "None")
        ]
        topologies = topologies.explode(ignore_index=True)
        topologies = _merge_lines(topologies)
        buffered = {}
        for _, row in topologies.iterrows():
            _buffer_sz(row, buffer_distance, topologies.crs, out=buffered)
        buffered = gpd.GeoDataFrame(buffered, geometry="geometry", crs=topologies.crs)

        if clip_to_overriding_plate:
            clipped = []
            for plate_id in buffered["over"].unique():
                try:
                    poly_match = plate_polygons[
                        plate_polygons["reconstruction_plate_ID"] == plate_id
                    ]
                    if poly_match.empty:
                        print(f"[WARN] No plate polygon match for plate_id: {plate_id}")
                        continue
    
                    intersection = gpd.overlay(
                        buffered[buffered["over"] == plate_id],
                        poly_match,
                    )
    
                    if not intersection.empty:
                        clipped.append(intersection)
    
                except Exception as e:
                    print(f"[ERROR] Clipping failed for plate_id {plate_id}: {e}")
    
            if clipped:
                clipped = gpd.GeoDataFrame(pd.concat(clipped, ignore_index=True))
                clipped = clipped[["name", "polarity", "feature_type", "over", "geometry"]]
                clipped = clipped.rename(columns={"over": "plate_id", "feature_type": "ftype"})
                buffered = gpd.GeoDataFrame(clipped, geometry="geometry")

    if not buffered.geometry.is_valid.all():
        buffered.geometry = buffered.buffer(0)

    if output_dir is not None:
        output_filename = os.path.join(output_dir, f"buffer_zones_{time:0.0f}Ma.geojson")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            buffered.to_file(output_filename)
            
    if return_output:
        return buffered
    
    return None


def _buffer_sz(row, distance_degrees, crs, out):
    
    geom = gpd.GeoSeries(row["geometry"], crs=crs)
    point = geom.representative_point()
    proj = f"+proj=aeqd +lat_0={point.y.iloc[0]} +lon_0={point.x.iloc[0]} +x_0=0 +y_0=0"
    projected = geom.to_crs(proj)

    distance_metres = np.deg2rad(distance_degrees) * EARTH_RADIUS * 1000.0
    direction = 1.0 if str(row["polarity"]).lower() == "left" else -1.0
    projected_buffered = projected.buffer(distance_metres * direction, single_sided=True)
    buffered = projected_buffered.to_crs(crs)
    geometry_out = buffered.iloc[0]
    
    # Skip bad geometries
    if not _has_enough_points(geometry_out):
        return out

    # Decompose MultiPolygon
    parts = list(geometry_out.geoms) if isinstance(geometry_out, MultiPolygon) else [geometry_out]
    
    geometries_out = []
    for part in parts:
        try:
            wrapped = wrap_geometries(part, central_meridian=0.0, tessellate_degrees=0.1)
            if isinstance(wrapped, (list, tuple)):
                geometries_out.extend(wrapped)
            elif wrapped is not None:
                geometries_out.append(wrapped)
        except Exception as e:
            print(f"[WARN] Failed to wrap geometry: {e}")
            continue

    if isinstance(geometries_out, BaseGeometry):
        geometries_out = [geometries_out]

    # Append results
    for i in geometries_out:
        for column_name in row.index:
            if column_name == "geometry":
                continue
            out.setdefault(column_name, []).append(row[column_name])
        out.setdefault("geometry", []).append(i)

    return out


def _has_enough_points(geometry, min_points=3):
    if geometry is None or geometry.is_empty:
        return False
    if isinstance(geometry, Polygon):
        return len(geometry.exterior.coords) >= min_points
    elif isinstance(geometry, MultiPolygon):
        return any(len(poly.exterior.coords) >= min_points for poly in geometry.geoms)
    return False


def _extract_overriding_plates(
    time,
    topology_features,
    rotation_model,
    anchor_plate_id,
):
    
    resolved_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        [],  # Discard boundaries/networks
        float(time),
        resolved_sections,
        anchor_plate_id,
    )

    # Ignore flat slab topologies
    slab_types = {
        pygplates.FeatureType.gpml_slab_edge,
        pygplates.FeatureType.gpml_topological_slab_boundary,
    }
    resolved_sections = [
        i
        for i in resolved_sections
        if i.get_topological_section_feature().get_feature_type()
        not in slab_types
    ]

    geometries = []
    polarities = []
    names = []
    feature_types = []
    feature_ids = []
    plate_ids = []
    overriding_plates = []
    subducting_plates = []
    left_plates = []
    right_plates = []
    shared_1s = []
    shared_2s = []
    for i in resolved_sections:
        for segment in i.get_shared_sub_segments():
            geometry = segment.get_resolved_geometry()
            geometry = pygplates_to_shapely(geometry, tessellate_degrees=0.1)

            polarity = segment.get_feature().get_enumeration(
                pygplates.PropertyName.gpml_subduction_polarity,
                "None",
            )
            if polarity == "Unknown":
                polarity = "None"
            valid_polarities = {"None", "Left", "Right"}
            if polarity not in valid_polarities:
                warnings.warn(
                    "Unknown polarity: {}".format(polarity), RuntimeWarning
                )
                continue

            name = segment.get_feature().get_name()
            if "flat slab" in name.lower():
                continue

            feature_type = (
                segment.get_feature().get_feature_type().to_qualified_string()
            )
            feature_id = segment.get_feature().get_feature_id().get_string()
            plate_id = segment.get_feature().get_reconstruction_plate_id(-1)
            tmp = segment.get_overriding_and_subducting_plates()
            if tmp is None:
                overriding_plate = -1
                subducting_plate = -1
            else:
                overriding_plate, subducting_plate = tmp
                overriding_plate = (
                    overriding_plate.get_feature().get_reconstruction_plate_id(
                        -1
                    )
                )
                subducting_plate = (
                    subducting_plate.get_feature().get_reconstruction_plate_id(
                        -1
                    )
                )
            del tmp
            left_plate = segment.get_feature().get_left_plate(-1)
            right_plate = segment.get_feature().get_right_plate(-1)

            sharing_topologies = segment.get_sharing_resolved_topologies()
            if len(sharing_topologies) > 0:
                shared_1 = (
                    sharing_topologies[0]
                    .get_feature()
                    .get_reconstruction_plate_id(-1)
                )
            else:
                shared_1 = -1
            if len(sharing_topologies) > 1:
                shared_2 = (
                    sharing_topologies[1]
                    .get_feature()
                    .get_reconstruction_plate_id(-1)
                )
            else:
                shared_2 = -1

            geometries.append(geometry)
            polarities.append(polarity)
            names.append(name)
            feature_types.append(feature_type)
            feature_ids.append(feature_id)
            plate_ids.append(plate_id)
            overriding_plates.append(overriding_plate)
            subducting_plates.append(subducting_plate)
            left_plates.append(left_plate)
            right_plates.append(right_plate)
            shared_1s.append(shared_1)
            shared_2s.append(shared_2)

    gdf = gpd.GeoDataFrame(
        {
            "polarity": polarities,
            "geometry": geometries,
            "name": names,
            "type": feature_types,
            "id": feature_ids,
            "plate_id": plate_ids,
            "over": overriding_plates,
            "subd": subducting_plates,
            "left": left_plates,
            "right": right_plates,
            "shared_1": shared_1s,
            "shared_2": shared_2s,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    
    return gdf


def _merge_lines(
    data: gpd.GeoDataFrame,
    groupby: Iterable[Hashable] = ("polarity", "type", "over"),
):
    
    out = []
    for gb_vals, grouped in data.groupby(list(groupby)):
        geom = linemerge(grouped.geometry.to_list())
        if isinstance(geom, BaseMultipartGeometry):
            geom = list(geom.geoms)
        else:
            geom = [geom]
        gb_data = {
            "geometry": geom,
            **{
                gb_col: gb_val
                for gb_col, gb_val
                in zip(groupby, gb_vals)
            }
        }
        if "name" not in gb_data.keys():
            gb_data["name"] = ":".join(grouped["name"].unique())
        out.append(
            gpd.GeoDataFrame(gb_data, geometry="geometry")
        )
    out = gpd.GeoDataFrame(
        pd.concat(out, ignore_index=True),
        geometry="geometry",
        crs=data.crs,
    )
    
    return out
