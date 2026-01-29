'''
Grid Visualisation

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 16/09/2025
'''


import glob
import os
import subprocess

import cartopy.crs as ccrs
import cmcrameri.cm as ccm
import dask.array as da
from joblib import Parallel, delayed
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt
from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import netCDF4 as nc
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
import xarray as xr

from gplately import (
    grids,
    PlateReconstruction,
    PlotTopologies,
)


def find_min_max(folder_path, variable_name="z"):

    # Find all NetCDF files in the specified folder
    nc_files = glob.glob(os.path.join(folder_path, "*.nc"))
    
    if not nc_files:
        print(f"No NetCDF files found in {folder_path}")
        return None, None
    
    # Initialize min and max values
    overall_min = float("inf")
    overall_max = float("-inf")
    
    # Process each file
    for file_path in tqdm(nc_files):
        try:
            # Open the NetCDF file
            with nc.Dataset(file_path, "r") as dataset:
                # Check if the variable exists in this file
                if variable_name in dataset.variables:
                    # Get the variable data
                    z_data = dataset.variables[variable_name][:]
                    
                    # Handle missing values or fill values
                    if hasattr(dataset.variables[variable_name], "_FillValue"):
                        fill_value = dataset.variables[variable_name]._FillValue
                        z_data = np.ma.masked_equal(z_data, fill_value)
                    
                    # Update min and max values
                    file_min = np.nanmin(z_data)
                    file_max = np.nanmax(z_data)
                    
                    overall_min = min(overall_min, file_min)
                    overall_max = max(overall_max, file_max)
                    
                else:
                    print(f"Variable '{variable_name}' not found in {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
    
    # Check if any valid data was found
    if overall_min == float("inf") or overall_max == float("-inf"):
        print(f"No valid data found for variable '{variable_name}'")
        return None, None
    
    return overall_min, overall_max


def find_percentiles(folder_path, variable_name="z", percentiles=(1, 99)):
    
    # Find all NetCDF files in the specified folder
    nc_files = glob.glob(os.path.join(folder_path, "*.nc"))

    if not nc_files:
        print(f"No NetCDF files found in {folder_path}")
        return None, None

    # List to hold all data
    all_data = []

    # Process each file
    for file_path in tqdm(nc_files):
        try:
            with nc.Dataset(file_path, "r") as dataset:
                if variable_name in dataset.variables:
                    z_data = dataset.variables[variable_name][:]

                    # Handle fill or missing values
                    if hasattr(dataset.variables[variable_name], "_FillValue"):
                        fill_value = dataset.variables[variable_name]._FillValue
                        z_data = np.ma.masked_equal(z_data, fill_value)

                    # Flatten and filter out masked values
                    z_data_flat = z_data.compressed() if isinstance(z_data, np.ma.MaskedArray) else z_data.flatten()
                    all_data.append(z_data_flat)
                else:
                    print(f"Variable '{variable_name}' not found in {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    if not all_data:
        print(f"No valid data found for variable '{variable_name}'")
        return None, None

    # Concatenate all data and compute percentiles
    combined_data = np.concatenate(all_data)
    p1, p99 = np.nanpercentile(combined_data, percentiles)

    return p1, p99


def find_percentiles_dask(folder_path, variable_name="z", percentiles=(1, 99)):
    
    # Find all NetCDF files in the specified folder
    nc_files = glob.glob(os.path.join(folder_path, "*.nc"))
    
    if not nc_files:
        print(f"No NetCDF files found in {folder_path}")
        return None, None
    
    # List to hold dask arrays
    dask_arrays = []
    
    # Process each file
    for file_path in tqdm(nc_files):
        try:
            # Open with xarray
            ds = xr.open_dataset(file_path, chunks='auto')
            
            if variable_name in ds.variables:
                z_data = ds[variable_name]
                
                # Convert to dask array and handle masked values
                z_dask = z_data.data
                
                # Flatten the array
                z_flat = z_dask.flatten()
                
                # Filter out NaN and fill values
                z_valid = z_flat[da.isfinite(z_flat)]
                
                dask_arrays.append(z_valid)
            else:
                print(f"Variable '{variable_name}' not found in {os.path.basename(file_path)}")
                ds.close()
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
    
    if not dask_arrays:
        print(f"No valid data found for variable '{variable_name}'")
        return None, None
    
    # Concatenate all dask arrays
    combined_data = da.concatenate(dask_arrays)
    
    # Compute percentiles using Dask's approximate percentile algorithm
    # This processes data in chunks without loading everything into memory
    p1, p99 = da.percentile(combined_data, percentiles).compute()
    
    return p1, p99


def _generate_grd_map(gplot, grd_dir, grd_filename, cb_label, vmin, vmax, projection, time, output_dir, reverse_lat):
    
    grd_filename_ =  grd_filename + f"_{time}Ma.nc"
    grd_file = os.path.join(grd_dir, grd_filename_)
    
    grd = grids.read_netcdf_grid(grd_file)
    
    if reverse_lat:
        grd = grd[::-1, :]

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=projection, facecolor=(plt.cm.colors.to_rgba("darkgray", alpha=0.5)))
    ax.set_global()

    im = gplot.plot_grid(ax, grd.data, cmap=ccm.lapaz_r, vmin=vmin, vmax=vmax, alpha=0.7, zorder=1)
    
    gplot.plot_coastlines(ax, facecolor="darkgray", edgecolor="none", zorder=2)
    gplot.plot_plate_motion_vectors(ax, spacingX=10, spacingY=10, normalise=True, alpha=0.1, zorder=3)
    gplot.plot_topological_plate_boundaries(ax, color="dimgray", linewidth=1.5, zorder=4)
    gplot.plot_trenches(ax, color="black", zorder=5)
    gplot.plot_subduction_teeth(ax, spacing=0.05, color="black", zorder=6)

    # Mollweide
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, linewidth=1, color="gray", alpha=0.3, linestyle="--", zorder=7)
    
    # Robinson
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     # Only x labels on bottom, y labels on left:
    #     draw_labels={"bottom": "x", "left": "y"},
    #     x_inline=False,
    #     linewidth=1,
    #     color="gray",
    #     alpha=0.3,
    #     linestyle="--",
    #     zorder=7,
    # )
    
    # Mollweide
    gl.top_labels=False
    gl.bottom_labels=False
    # gl.right_labels=False
    # gl.left_labels=False
    
    gl.xlabel_style = {"size": 16}
    gl.ylabel_style = {"size": 16}
    
    # Mollweide
    ax.text(0.49,-0.03, "60°E", transform=ax.transAxes, fontsize=16)
    ax.text(0.46,-0.03, "0°", transform=ax.transAxes, fontsize=16)
    ax.text(0.40,-0.025, "60°W", transform=ax.transAxes, fontsize=16)
    
    # Robinson
    # ax.text(0.39,-0.03, "60°E", transform=ax.transAxes, fontsize=16)
    # ax.text(0.49,-0.03, "0°", transform=ax.transAxes, fontsize=16)
    # ax.text(0.56,-0.03, "60°W", transform=ax.transAxes, fontsize=16)
    
    cb = fig.colorbar(im, orientation="horizontal", shrink=0.4, pad=0.06, extend="max")
    cb.set_label(cb_label, fontsize=16, labelpad=10)
    # cb.set_ticks([0, 5000, 10000, 15000])
    cb.ax.tick_params(labelsize=16)
    
    trench_handle = Line2D([], [], color='black')
    
    custom_handles = [
        Patch(facecolor="darkgray", edgecolor="none"),
        trench_handle,
        Line2D([0], [0], color="dimgray", lw=2),
    ]
    custom_labels = [
        "Continental Crust",
        "Trench Lines",
        "Mid-Ocean Ridges/\nTransform Faults",
    ]
    
    ax.legend(custom_handles, custom_labels, fontsize=16, loc="lower left", bbox_to_anchor=(0, -0.25),
              handler_map={trench_handle: HandlerTrenchLine()})

    ax.set_title(f"{time} Ma", fontsize=25, y=1.04)

    filename = os.path.join(output_dir, grd_filename + f"_map_{time:.0f}Ma.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def _generate_grd_maps_parallel(
    rotation_model,
    topology_features,
    static_polygons,
    coastlines,
    continents,
    COBs,
    grd_dir,
    grd_filename,
    cb_label,
    vmin,
    vmax,
    projection,
    time,
    output_dir,
    reverse_lat,
    anchor_plate_id,
    ):
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )
    
    gplot = PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        anchor_plate_id=anchor_plate_id,
        time=time,
    )
    
    return _generate_grd_map(
        gplot,
        grd_dir,
        grd_filename,
        cb_label,
        vmin,
        vmax,
        projection,
        time,
        output_dir,
        reverse_lat,
    )


def generate_grd_maps(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        grd_dir,
        grd_filename,
        cb_label,
        vmin,
        vmax,
        projection,
        time_steps,
        output_dir,
        reverse_lat=False,
        anchor_plate_id=0,
        n_jobs=-2
        ):
    
    tasks = (delayed(_generate_grd_maps_parallel)(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        grd_dir,
        grd_filename,
        cb_label,
        vmin,
        vmax,
        projection,
        time,
        output_dir,
        reverse_lat,
        anchor_plate_id,
        ) for time in tqdm(time_steps, desc="Dispatching tasks"))
    
    Parallel(n_jobs=n_jobs, backend="loky")(tasks)


# Custom handler for trench line with triangles
class HandlerTrenchLine(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        # Horizontal line
        line = Line2D([xdescent, xdescent + width], 
                      [ydescent + height / 2] * 2, 
                      color='black', lw=2, transform=trans)

        # Taller, sharper triangle teeth
        def make_triangle(x_center):
            base_width = width * 0.05
            height_offset = height * 0.3
            return Polygon([
                [x_center, ydescent + height / 2 + height_offset],  # tip
                [x_center - base_width, ydescent + height / 2],     # left base
                [x_center + base_width, ydescent + height / 2],     # right base
            ], closed=True, color='black', transform=trans)

        # Two teeth positioned proportionally
        triangle1 = make_triangle(xdescent + width * 0.35)
        triangle2 = make_triangle(xdescent + width * 0.7)

        return [line, triangle1, triangle2]
    

def create_animation(
    image_filenames,
    output_filename,
    fps=5,
    codec="auto",
    bitrate="5000k",
    output_fps=30,
    ffmpeg_params=None,
    **kwargs
):
    
    if codec == "hevc":
        if hwaccel_available():
            codec = "hevc_videotoolbox"
        else:
            codec = "hevc"
    elif codec == "auto":
        codec = "libx264"

    if ffmpeg_params is None:
        ffmpeg_params = [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ]

    logger = kwargs.pop("logger", None)
    audio = kwargs.pop("audio", False)

    with ImageSequenceClip(image_filenames, fps=fps) as clip:
        clip.write_videofile(
            output_filename,
            fps=output_fps,
            codec=codec,
            bitrate=bitrate,
            audio=audio,
            logger=logger,
            ffmpeg_params=ffmpeg_params,
            **kwargs,
        )


def hwaccel_available(codec="hevc_videotoolbox"):
    
    return codec_available(codec)


def codec_available(codec):
    
    result = _test_codec(codec)
    
    return result.returncode == 0

def _test_codec(codec):
    
    cmd = [
        FFMPEG_BINARY,
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=color=black:size=1080x1080",
        "-vframes", "1",
        "-pix_fmt", "yuv420p10le",
        "-an",
        "-c:v", codec,
        "-f", "null",
        "-",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    
    return result
