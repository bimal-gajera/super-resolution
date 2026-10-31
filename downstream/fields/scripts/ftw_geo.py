"""Geometry shared by the Fields of The World (FTW) scripts.

An FTW chip is 256 x 256 px in EPSG:4326 at 5.4e-5 degrees per pixel (about 6 m north-south and 3-6 m east-west):
Sentinel-2 interpolated onto a finer grid, ~1.5 km north-south by 0.7-1.5 km east-west. PixelDiT takes 48 x 48 px
windows of the 10 m UTM grid, so every chip gets a working grid: its bounding box in its UTM zone, snapped to
multiples of 10 m (where Sentinel-2 pixels lie). Every product of a chip (the 10 m input, the 1 m mosaic of the SR
windows, the label rasters) lives on that grid at some resolution; `Grid.transform(res)` georeferences it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
LR_PATCH = 48      # PixelDiT's window, Sentinel-2 pixels
HR_SIZE = 1584     # PixelDiT's output per window (0.303 m)
S2_M = 10.0        # Sentinel-2 pixel, metres
WINDOWS = ("window_a", "window_b")   # FTW's two dates per chip (start and end of the season)


@dataclass(frozen=True)
class Grid:
    epsg: int
    x_min: float   # west edge, metres
    y_max: float   # north edge, metres
    h: int         # rows of 10 m pixels
    w: int         # columns of 10 m pixels

    def transform(self, res: float = S2_M) -> Affine:
        return Affine(res, 0, self.x_min, 0, -res, self.y_max)

    def shape(self, res: float = S2_M) -> tuple[int, int]:
        return int(round(self.h * S2_M / res)), int(round(self.w * S2_M / res))

    def bounds(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_max - self.h * S2_M, self.x_min + self.w * S2_M, self.y_max


def read_selection(path: Path) -> list[tuple[str, str]]:
    """Lines of `<country> <aoi_id>`."""
    return [tuple(line.split()) for line in Path(path).read_text().splitlines() if line.strip()]


def chip_path(ftw: Path, country: str, aoi_id: str, kind: str) -> Path:
    """kind: window_a | window_b | instance | semantic_3class (FTW's own folder layout)."""
    sub = f"s2_images/{kind}" if kind in WINDOWS else f"label_masks/{kind}"
    return Path(ftw) / country / sub / f"{aoi_id}.tif"


def chip_grid(path: Path) -> Grid:
    with rasterio.open(path) as src:
        # FTW rasters are stored south-up (row 0 is the southern edge), so src.bounds has top and bottom swapped
        west, east = sorted((src.transform.c, src.transform.c + src.transform.a * src.width))
        south, north = sorted((src.transform.f, src.transform.f + src.transform.e * src.height))
        lon, lat = (west + east) / 2, (south + north) / 2
        epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
        x0, y0, x1, y1 = transform_bounds(src.crs, epsg, west, south, east, north, densify_pts=21)
    x0, y0 = np.floor(x0 / S2_M) * S2_M, np.floor(y0 / S2_M) * S2_M
    x1, y1 = np.ceil(x1 / S2_M) * S2_M, np.ceil(y1 / S2_M) * S2_M
    h, w = int(round((y1 - y0) / S2_M)), int(round((x1 - x0) / S2_M))
    return Grid(epsg, float(x0), float(y1), max(h, LR_PATCH), max(w, LR_PATCH))


def read_lr(path: Path, grid: Grid, bands: tuple[int, ...] = (1, 2, 3)) -> tuple[np.ndarray, np.ndarray]:
    """The chip on its 10 m grid: (bands, h, w) float32 DN and the mask of pixels inside the chip.

    Bands 1-3 of an FTW chip are B04, B03, B02 (red, green, blue). Bilinear sampling at the 10 m pixel centres;
    the corners of the grid that fall outside the chip (the chip is a lon / lat box, slightly skewed in UTM) are
    filled with the nearest value inside so that every window is complete.
    """
    with rasterio.open(path) as src:
        out = np.full((len(bands), grid.h, grid.w), np.nan, np.float32)
        reproject(src.read(bands).astype(np.float32), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid.transform(), dst_crs=grid.epsg, resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
    inside = np.isfinite(out).all(0)
    if not inside.all():
        idx = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
        out = out[:, idx[0], idx[1]]
    return out, inside


def footprint(path: Path, grid: Grid, res: float) -> np.ndarray:
    """Mask of the grid at `res` metres that lies inside the chip."""
    with rasterio.open(path) as src:
        out = np.zeros(grid.shape(res), np.uint8)
        reproject(np.ones((src.height, src.width), np.uint8), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid.transform(res), dst_crs=grid.epsg, resampling=Resampling.nearest)
    return out.astype(bool)


def window_offsets(length: int, size: int = LR_PATCH, min_overlap: int = 8) -> list[int]:
    """Evenly spaced window starts covering `length` pixels with at least `min_overlap` pixels of overlap."""
    if length <= size:
        return [0]
    n = int(np.ceil((length - min_overlap) / (size - min_overlap)))
    return [int(round(v)) for v in np.linspace(0, length - size, n)]


def chip_windows(grid: Grid) -> list[tuple[int, int]]:
    return [(r, c) for r in window_offsets(grid.h) for c in window_offsets(grid.w)]


def area_resize(image: np.ndarray, size: int) -> np.ndarray:
    """Exact box-filter resampling of a square (n, n[, channels]) image to size x size (fractional boxes)."""
    image = np.asarray(image, np.float64)
    n = image.shape[0]
    edges = np.linspace(0, n, size + 1)
    lo = np.floor(edges).astype(int)
    frac = edges - lo
    hi = np.minimum(lo + 1, n)
    for axis in (0, 1):
        c = np.concatenate([np.zeros_like(np.take(image, [0], axis)), np.cumsum(image, axis)], axis)
        shape = [1] * image.ndim
        shape[axis] = -1
        v = np.take(c, lo, axis) * (1 - frac).reshape(shape) + np.take(c, hi, axis) * frac.reshape(shape)
        image = np.diff(v, axis=axis)
    return image / (n / size) ** 2


def block_mean(image: np.ndarray, k: int) -> np.ndarray:
    """Average k x k blocks (the 1 m mosaic -> 3 m with k = 3); trailing rows / columns that do not fill a block
    are dropped."""
    h, w = image.shape[0] // k * k, image.shape[1] // k * k
    return image[:h, :w].reshape(h // k, k, w // k, k, *image.shape[2:]).mean((1, 3))


_POLYGONS: dict[Path, gpd.GeoDataFrame] = {}


def load_polygons(folder: Path, country: str, grid: Grid) -> gpd.GeoSeries:
    """The FTW field polygons that touch the grid, in the grid's UTM zone (boundaries_<country>_<year>.parquet)."""
    path = next(Path(folder).glob(f"boundaries_{country}_*.parquet"))
    if path not in _POLYGONS:
        _POLYGONS[path] = gpd.read_parquet(path, columns=["geometry"])
    polys = _POLYGONS[path]
    area = gpd.GeoSeries([box(*grid.bounds())], crs=grid.epsg).to_crs(polys.crs).iloc[0]
    hit = polys.iloc[polys.sindex.query(area, predicate="intersects")]
    return hit.geometry.to_crs(grid.epsg).make_valid()


def label_rasters(polys: gpd.GeoSeries, grid: Grid, res: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polygon outlines, field mask and distance to the nearest outline (metres) on the grid at `res` metres."""
    shape, transform = grid.shape(res), grid.transform(res)
    geoms = [g for g in polys if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, bool), np.zeros(shape, bool), np.full(shape, np.inf, np.float32)
    line = rasterize([g.boundary for g in geoms], out_shape=shape, transform=transform, dtype=np.uint8).astype(bool)
    field = rasterize(geoms, out_shape=shape, transform=transform, dtype=np.uint8).astype(bool)
    dist = (ndimage.distance_transform_edt(~line) * res).astype(np.float32)
    return line, field, dist


def chip_offset(ftw: Path, country: str, aoi_id: str) -> int:
    """1000 if the chip carries the +1000 DN offset of Sentinel-2 L2A products of 2022 and later, else 0 (a
    1.5 km chip always has dark pixels, so no red, green or blue value below 950 means the offset is there)."""
    low = []
    for window in WINDOWS:
        with rasterio.open(chip_path(ftw, country, aoi_id, window)) as src:
            low.append(int(src.read((1, 2, 3)).min()))
    return 1000 if min(low) >= 950 else 0


def chip_footprint(path: Path) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a chip in degrees (FTW rasters are stored south-up, see chip_grid)."""
    with rasterio.open(path) as src:
        west, east = sorted((src.transform.c, src.transform.c + src.transform.a * src.width))
        south, north = sorted((src.transform.f, src.transform.f + src.transform.e * src.height))
    return west, south, east, north
