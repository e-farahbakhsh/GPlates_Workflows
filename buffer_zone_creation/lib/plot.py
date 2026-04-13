import os
import subprocess

import cartopy.crs as ccrs
import geopandas as gpd
from joblib import Parallel, delayed
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt
from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm

from gplately import (
    PlateReconstruction,
    PlotTopologies,
)


def _generate_buffer_zone_map(gplot, projection, time, buffer_zones_dir, output_dir):
    
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=projection, facecolor="azure")
    # ax.set_global()
    ax.set_extent([30, 90, 15, 45], crs=ccrs.PlateCarree())

    gplot.plot_continents(ax, edgecolor="none", facecolor="tan", alpha=0.5, zorder=1)
    gplot.plot_coastlines(ax, edgecolor="none", facecolor="tan", alpha=0.7, zorder=2)
    gplot.plot_plate_motion_vectors(ax, spacingX=10, spacingY=10, normalise=True, alpha=0.1, zorder=3)

    buffer_zones_t_filename = f'buffer_zones_{time}Ma.geojson'
    buffer_zones_t_filename = os.path.join(buffer_zones_dir, buffer_zones_t_filename)
    buffer_zones_t = gpd.read_file(buffer_zones_t_filename)

    buffer_zones_t.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        facecolor='palegreen',
        edgecolor='none',
        alpha=0.7,
        zorder=4,
    )

    gplot.plot_topological_plate_boundaries(ax, color="dimgray", linewidth=1.2, zorder=5)
    gplot.plot_trenches(ax, color="black", zorder=6)
    gplot.plot_subduction_teeth(ax, spacing=0.015, color="black", zorder=7)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False,
                      linewidth=1, color="gray", alpha=0.3, linestyle="--", zorder=8)
    
    gl.top_labels=False
    # gl.bottom_labels=False
    gl.right_labels=False
    # gl.left_labels=False    
    
    gl.xlabel_style = {"size": 16}
    gl.ylabel_style = {"size": 16}

    # ax.text(0.49,-0.03, "60°E", transform=ax.transAxes, fontsize=16)
    # ax.text(0.46,-0.03, "0°", transform=ax.transAxes, fontsize=16)
    # ax.text(0.40,-0.025, "60°W", transform=ax.transAxes, fontsize=16)

    # Dummy handle to trigger custom handler
    trench_handle = Line2D([], [], color="black")
    
    # Add custom handles
    custom_handles = [
        Patch(facecolor="tan", edgecolor="none"),
        Patch(facecolor="palegreen", edgecolor="none", alpha=0.7),
        trench_handle,
        Line2D([0], [0], color="orangered", lw=2),
    ]
               
    custom_labels = [
        "Continental Crust",
        "Target Arc Environment",
        "Subduction zone",
        "Plate boundary",
    ]

    # Draw legend
    ax.legend(custom_handles, custom_labels, fontsize=16, loc="lower left", bbox_to_anchor=(0, -0.3),
              handler_map={trench_handle: HandlerTrenchLine()})

    ax.set_title(f"{time} Ma", fontsize=25, y=1.04)

    filename = os.path.join(output_dir, f"buffer_zone_map_{time:.0f}Ma.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def _generate_buffer_zone_maps_parallel(
    rotation_model,
    topology_features,
    static_polygons,
    coastlines,
    continents,
    COBs,
    projection,
    time,
    buffer_zones_dir,
    output_dir,
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
    
    return _generate_buffer_zone_map(
        gplot,
        projection,
        time,
        buffer_zones_dir,
        output_dir,
    )


def generate_buffer_zone_maps(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        projection,
        time_steps,
        buffer_zones_dir,
        output_dir,
        anchor_plate_id=0,
        n_jobs=-2
        ):
    
    tasks = (delayed(_generate_buffer_zone_maps_parallel)(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        projection,
        time,
        buffer_zones_dir,
        output_dir,
        anchor_plate_id,
        ) for time in tqdm(time_steps, desc="Dispatching tasks"))
    
    Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    

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


def format_feature_name(s, bold=False):
    
    """Make feature names easier to read in plots."""
    s = s.replace("_", " ")
    s = s[0].capitalize() + s[1:]

    replace = {
        "(cm/yr)": r"($\mathrm{cm \; {yr}^{-1}}$)",
        "(m)": r"($\mathrm{m}$)",
        "(m^3/m^2)": r"($\mathrm{m^3 \; m^{-2}}$)",
        "(m^2/yr)": r"($\mathrm{m^2 \; {yr}^{-1}}$)",
        "(t/m^2)": r"($\mathrm{t \; m^{-2}}$)",
        "(Ma)": r"($\mathrm{Ma}$)",
        "(degrees)": r"($\mathrm{\degree}$)",
        "(km)": r"($\mathrm{km}$)",
        "(km/Myr)": r"($\mathrm{km \; {Myr}^{-1}}$)",
        "(/Ps)": r"($\mathrm{{Ps}^{-1}}$)",
        "(/s)": r"($\mathrm{{s}^{-1}}$)",
        "(rad/Ps)": r"($\mathrm{rad. \; {Ps}^{-1}}$)",
    }
    if bold:
        replace = {
            key: value.replace(r"\mathrm", r"\mathbf")
            for key, value in replace.items()
        }
    for key, value in replace.items():
        s = s.replace(key, value)

    return s


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
