#!/usr/bin/env python3
"""Second pass of scripts/ftw12_chips.py: chips whose 12-band file does not reproduce the published FTW chip.

Since FTW built its chips, Planetary Computer has added reprocessed copies of some scenes and updated cloud-cover
values, so FTW's selection rule can now land on another copy or another date of the same season window. FTW's
original scene is still among the rule's candidates. This pass goes through those candidates (FTW's search, FTW's
order) and takes the one whose B04 B03 B02 B08, made with FTW's loading steps, equal the published chip
(>= 99 % identical values, as in the first pass), then writes its 12 bands with the same steps. Candidates that
worked for the previous chip of the same country and window are tried first (neighbouring chips share scenes).

    python scripts/ftw12_repair.py --workers 12

Reads data/ftw12/scenes/*.csv (first pass), rewrites the affected data/ftw12/<country>/s2_images/<window>/<aoi>.tif,
and writes data/ftw12/scenes_repair/<task>.csv (scene id, search = "identified" or "not found"). Resumable: chips
already in scenes_repair are skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client

from ftw12_chips import FTW_BANDS, MATCH, ROOT, SOURCE, compare, fetch, season_window, write_chip
from download_s2_chips import check_if_scene_contains_box, get_scene_for_box  # FTW's own (path set by ftw12_chips)


def four_bands(item, box) -> np.ndarray:
    """B04 B03 B02 B08 of a candidate on the chip's grid, with FTW's loading steps."""
    d = odc.stac.stac_load([planetary_computer.sign(item)], bands=FTW_BANDS, bbox=tuple(box.bounds), crs="EPSG:4326",
                           dtype="uint16", resolution=0.00009009, resampling="bilinear").isel(time=0)
    d = d.interp(longitude=np.linspace(float(d.longitude.min()), float(d.longitude.max()), 256),
                 latitude=np.linspace(float(d.latitude.min()), float(d.latitude.max()), 256))
    return np.stack([d[b].values for b in FTW_BANDS]).astype(np.int64)


def run_task(task: dict) -> str:
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    out_dir, country, window = Path(task["out"]), task["country"], task["window"]
    rows, recent = [], []                                   # recent: scenes identified for earlier chips
    for aoi_id, box_wkb in task["chips"]:
        box = gpd.GeoSeries.from_wkb([box_wkb]).iloc[0]
        published = Path(task["ftw"]) / country / "s2_images" / window / f"{aoi_id}.tif"
        out = out_dir / country / "s2_images" / window / f"{aoi_id}.tif"
        row = dict(country=country, aoi_id=aoi_id, window=window, scene_id=None, search="not found", tried=0,
                   equal_share=None, error=None)
        for attempt in range(3):
            try:
                if not fetch(f"{SOURCE}/{country}/s2_images/{window}/{aoi_id}.tif", published):
                    row["error"] = "no published FTW chip"
                    break
                import rasterio
                with rasterio.open(published) as f:
                    pub = f.read().astype(np.int64)
                cands = get_scene_for_box(catalog, box, task["datetimes"][aoi_id])
                first = [it for it in recent if check_if_scene_contains_box(it, box)]
                order = first + [it for it in cands if it.id not in {r.id for r in first}]
                for k, item in enumerate(order, 1):
                    x = four_bands(item, box)
                    if x.shape == pub.shape and (x == pub).mean() >= MATCH:
                        write_chip(planetary_computer.sign(item), box, out)
                        row.update(scene_id=item.id, search="identified", tried=k, **{
                            k2: v for k2, v in compare(out, published).items() if k2 == "equal_share"})
                        recent = [item] + [r for r in recent if r.id != item.id][:4]
                        break
                else:
                    row["tried"] = len(order)
                row["error"] = None
                break
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]
                time.sleep(5 * (attempt + 1))
        rows.append(row)
    path = out_dir / "scenes_repair" / f"{task['name']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    found = sum(r["search"] == "identified" for r in rows)
    return f"{task['name']}: {found} of {len(rows)} identified"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ftw", type=Path, default=ROOT / "data/ftw")
    ap.add_argument("--configs", type=Path, default=ROOT / "data/ftw_configs")
    ap.add_argument("--out", type=Path, default=ROOT / "data/ftw12")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="only this many chips (for a check)")
    ap.add_argument("--one-per-country", action="store_true", help="with --limit: spread the check over countries")
    args = ap.parse_args()

    first = pd.concat([pd.read_csv(p, dtype={"aoi_id": str}) for p in glob.glob(str(args.out / "scenes/*.csv"))])
    first = first.drop_duplicates(["country", "aoi_id", "window"], keep="last")
    todo = first[first.scene_id.isna() | (first.equal_share.fillna(0) < MATCH)]
    done = glob.glob(str(args.out / "scenes_repair/*.csv"))
    if done:
        d = pd.concat([pd.read_csv(p, dtype={"aoi_id": str}) for p in done])
        keys = set(zip(d.country, d.aoi_id, d.window))
        todo = todo[[k not in keys for k in zip(todo.country, todo.aoi_id, todo.window)]]
    if args.limit:
        todo = (todo.groupby("country").head(1) if args.one_per_country else todo).head(args.limit)
    tasks = []
    for (country, window), g in todo.groupby(["country", "window"]):
        cfg = json.loads((args.configs / f"data_config_{country}.json").read_text())
        chips = gpd.read_parquet(ROOT / f"data/ftw_chips/chips_{country}.parquet")
        chips["aoi_id"] = chips.aoi_id.astype(str)
        chips = chips[chips.aoi_id.isin(set(g.aoi_id))]            # FTW's table order
        part_all = [(a, geom.wkb) for a, geom in zip(chips.aoi_id, chips.geometry)]
        for k in range(0, len(part_all), args.chunk):
            part = part_all[k : k + args.chunk]
            tasks.append(dict(name=f"{country}_{window}_{k // args.chunk:04d}", country=country, window=window,
                              chips=part, datetimes={a: season_window(cfg, a, window) for a, _ in part},
                              out=str(args.out), ftw=str(args.ftw)))
    print(f"{len(todo)} chip-dates to repair in {len(tasks)} tasks with {args.workers} workers", flush=True)
    started, n = time.time(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(run_task, t) for t in tasks]):
            n += 1
            print(f"[{n}/{len(tasks)} tasks, {(time.time() - started) / 3600:.2f} h] {fut.result()}", flush=True)
    print("finished")


if __name__ == "__main__":
    main()
