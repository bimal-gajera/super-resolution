#!/usr/bin/env python3
"""FTW chips with all 12 Sentinel-2 bands, made with FTW's own chip-creation procedure.

FTW's official script (third_party/ftw-dataset-creation/download_s2_chips.py, github.com/fieldsoftheworld/
ftw-dataset-creation, commit 812de88) picks one Sentinel-2 L2A scene per chip and date and writes B04 B03 B02 B08.
This runs the same procedure, with FTW's own functions for the scene search and the in-chip cloud check, and only
the band list extended to the 12 bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12:
  - search Planetary Computer `sentinel-2-l2a` over the chip in its season window (data_config_<country>.json),
    scene cloud cover < 90 %, sorted by cloud cover, scenes that contain the chip (reusing the previous search's
    scenes while they contain the next chip, as FTW's script does);
  - take the first scene with < 10 % cloud inside the chip (scene classification layer, classes 8 and 9);
  - load the bands with odc-stac onto EPSG:4326 at 0.00009009 deg (bilinear), interpolate to 256 x 256, write.
Two technical changes, same numbers: FTW's `np.linspace(da.min(), ...)` fails with current xarray, so floats are
passed; Planetary Computer links are signed right before each read (they expire during long runs).

    python scripts/ftw12_chips.py --select data/selections/ftp_all.txt --workers 16

Output (resumable: chips whose file exists are skipped):
  data/ftw12/<country>/s2_images/<window_a|window_b>/<aoi_id>.tif   12 bands, uint16, raw L2A values as stored
  data/ftw12/scenes/<task>.csv   chip, window, scene id, cloud share in the chip, and the check against FTW
Check: the published FTW chip (fetched into data/ftw/ if missing) is compared on its four bands; equal_share is the
share of identical values. If the scene from the reused search does not reproduce the chip, a fresh search is tried.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rioxarray  # noqa: F401  (registers .rio, as in FTW's script)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/ftw-dataset-creation"))
from download_s2_chips import (  # noqa: E402  FTW's own functions
    PATCH_CLOUD_THRESHOLD_PCT, WindowEmpty, check_if_scene_contains_box, check_if_scene_is_cloudy_at_box,
    get_scene_for_box,
)

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
FTW_BANDS = ["B04", "B03", "B02", "B08"]           # the bands and order of the published FTW chips
SOURCE = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kerner-lab/fields-of-the-world"
MATCH = 0.99                                       # share of identical values that counts as the same chip


def season_window(cfg: dict, aoi_id: str, window: str) -> str:
    """The date range of a chip's window; per-grid seasons are matched by grid id prefix, as in FTW's script."""
    seasons = cfg["seasons"]
    if isinstance(seasons, dict):
        s = seasons[window]
    else:
        ids = [g["id"] for g in cfg["grids"]]
        s = seasons[next(k for k, gid in enumerate(ids) if aoi_id.startswith(gid))][window]
    return f"{s['start']}/{s['end']}"


def fetch(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{time.time_ns()}.part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def write_chip(item, box, out: Path) -> None:
    data = odc.stac.stac_load([item], bands=BANDS, bbox=tuple(box.bounds), crs="EPSG:4326", dtype="uint16",
                              resolution=0.00009009, resampling="bilinear").isel(time=0)
    data = data.interp(longitude=np.linspace(float(data.longitude.min()), float(data.longitude.max()), 256),
                       latitude=np.linspace(float(data.latitude.min()), float(data.latitude.max()), 256))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.tif")
    data.to_array("band").rio.to_raster(tmp, driver="GTiff", dtype="uint16", tiled=True, blockxsize=256,
                                        blockysize=256)
    with rasterio.open(tmp, "r+") as dst:
        for i, b in enumerate(BANDS, 1):
            dst.set_band_description(i, b)
        dst.update_tags(sentinel2_id=item.id)
    tmp.replace(out)


def select(candidates, box, out: Path):
    """FTW's rule: the first candidate (sorted by scene cloud cover) with < 10 % cloud inside the chip."""
    for item in candidates:
        signed = planetary_computer.sign(item)
        try:
            cloud = check_if_scene_is_cloudy_at_box(signed, box)
        except WindowEmpty:
            continue
        if cloud < PATCH_CLOUD_THRESHOLD_PCT:
            write_chip(signed, box, out)
            return item.id, float(cloud)
    return None, None


def compare(ours: Path, published: Path) -> dict:
    with rasterio.open(ours) as a, rasterio.open(published) as b:
        x = a.read([BANDS.index(n) + 1 for n in FTW_BANDS]).astype(np.int64)
        y = b.read().astype(np.int64)
        if x.shape != y.shape:
            return dict(same_grid=False, equal_share=0.0, max_diff=-1)
        same_grid = a.transform.almost_equals(b.transform)
    d = np.abs(x - y)
    return dict(same_grid=bool(same_grid), equal_share=float((d == 0).mean()), max_diff=int(d.max()))


def run_task(task: dict) -> str:
    """One country, one window, a run of chips in the order of FTW's chips table (FTW's search reuse works along it)."""
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")   # signed per read
    country, window, out_dir = task["country"], task["window"], Path(task["out"])
    rows, found = [], None
    for aoi_id, box_wkb in task["chips"]:
        box = gpd.GeoSeries.from_wkb([box_wkb]).iloc[0]
        out = out_dir / country / "s2_images" / window / f"{aoi_id}.tif"
        if out.exists():
            continue
        row = dict(country=country, aoi_id=aoi_id, window=window, scene_id=None, cloud_in_chip=None, search="reused",
                   same_grid=None, equal_share=None, max_diff=None, error=None)
        for attempt in range(3):
            try:
                cands = [it for it in (found or []) if check_if_scene_contains_box(it, box)]
                if not cands:
                    found = get_scene_for_box(catalog, box, task["datetimes"][aoi_id])
                    cands, row["search"] = list(found), "new"
                scene, cloud = select(cands, box, out)
                published = Path(task["ftw"]) / country / "s2_images" / window / f"{aoi_id}.tif"
                have = fetch(f"{SOURCE}/{country}/s2_images/{window}/{aoi_id}.tif", published)
                if scene and have:
                    check = compare(out, published)
                    if check["equal_share"] < MATCH and row["search"] == "reused":
                        out.unlink(missing_ok=True)            # retry with a fresh search for this chip
                        found = get_scene_for_box(catalog, box, task["datetimes"][aoi_id])
                        scene, cloud = select(list(found), box, out)
                        row["search"] = "new (retry)"
                        check = compare(out, published) if scene else check
                    row.update(check)
                row.update(scene_id=scene, cloud_in_chip=cloud, published=have, error=None)
                break
            except Exception as e:                                  # network hiccups: retry, then record
                row["error"] = f"{type(e).__name__}: {e}"[:300]
                time.sleep(5 * (attempt + 1))
        rows.append(row)
    path = out_dir / "scenes" / f"{task['name']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)
    return f"{task['name']}: {len(rows)} chips"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", type=Path, required=True, help="lines of `<country> <aoi_id>`")
    ap.add_argument("--ftw", type=Path, default=ROOT / "data/ftw")
    ap.add_argument("--configs", type=Path, default=ROOT / "data/ftw_configs")
    ap.add_argument("--out", type=Path, default=ROOT / "data/ftw12")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100, help="chips per task")
    args = ap.parse_args()

    wanted = pd.read_csv(args.select, sep=" ", names=["country", "aoi_id"], dtype=str)
    tasks = []
    for country, g in wanted.groupby("country"):
        cfg_path = args.configs / f"data_config_{country}.json"
        fetch(f"{SOURCE}/{country}/data_config_{country}.json", cfg_path)
        cfg = json.loads(cfg_path.read_text())
        chips = gpd.read_parquet(ROOT / f"data/ftw_chips/chips_{country}.parquet")
        chips["aoi_id"] = chips.aoi_id.astype(str)
        chips = chips[chips.aoi_id.isin(set(g.aoi_id))]                # FTW's table order is kept
        for window in ("window_a", "window_b"):
            todo = [(a, geom.wkb) for a, geom in zip(chips.aoi_id, chips.geometry)
                    if not (args.out / country / "s2_images" / window / f"{a}.tif").exists()]
            dts = {a: season_window(cfg, a, window) for a, _ in todo}
            for k in range(0, len(todo), args.chunk):
                part = todo[k : k + args.chunk]
                tasks.append(dict(name=f"{country}_{window}_{k // args.chunk:04d}", country=country, window=window,
                                  chips=part, datetimes={a: dts[a] for a, _ in part}, out=str(args.out),
                                  ftw=str(args.ftw)))
    n = sum(len(t["chips"]) for t in tasks)
    print(f"{len(wanted)} chips selected; {n} chip-dates to make in {len(tasks)} tasks with {args.workers} workers",
          flush=True)
    started, done = time.time(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(run_task, t) for t in tasks]):
            done += 1
            print(f"[{done}/{len(tasks)} tasks, {(time.time() - started) / 3600:.2f} h] {fut.result()}", flush=True)
    print("finished")


if __name__ == "__main__":
    main()
