#!/usr/bin/env python3
"""Put 3 m SR outputs of FTW chips into the data layout of the paper's code (row "3").

Inputs per chip and date (both of FTW's windows):
  - the SR image <sr>/<country>/<window>/<aoi_id>_SR3m.tif written by the user's predict_ldm.py / predict_pixeldit.py
    from the 12-band FTW chip (RGB, uint8, the chip's grid with 10/3 times more pixels per side: 853 x 853);
  - NIR is added with the user's stack_nir.py, unchanged: Sentinel-2 B08 of the 12-band chip, bilinear onto the SR
    grid. Output: 4 bands, uint16 (R, G, B as the SR's 0-255 values, B08 raw L2A), read by the paper's code as it is.
Labels per chip, as FTP makes them (Sec. 3): the FTW polygons clipped to the chip footprint (chips_<country>.parquet)
and rasterised on the SR grid, 0 background, 1 field, 2 boundary; the boundary class is each polygon outline buffered
by one pixel ("buffering 3 m (about 1 pixel per side)").

    python scripts/make_sr3m_layout.py --sr data/ldm3m/sr --name ldm3m --train data/selections/ftp_train.txt \
        --val data/selections/ftp_val.txt --test data/selections/ftp_test.txt --workers 4

Writes data/ftp_layout/<name>/planet/<country>/{window_a,window_b,labels}/<aoi_id>.tif, planet/index.parquet and
ftw/<country>/chips_<country>.parquet (the layout ftw_planet's FTWPlanet reads), and with --panels N, N panels per
country in results/<name>_inputs/: FTW's Sentinel-2 | SR | SR with the label outlines | labels. Chips whose SR is not
there yet are skipped and counted; rerunning adds them (written files are kept).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parent.parent
INFERENCE = ROOT.parents[2] / "inference"           # the user's 3 m inference scripts
WINDOWS = ("window_a", "window_b")
CLASS_RGB = np.array([[40, 30, 25], [190, 230, 140], [255, 255, 255]], np.uint8)
_POLYS: dict[str, gpd.GeoDataFrame] = {}


def polygons(country: str) -> gpd.GeoDataFrame:
    if country not in _POLYS:
        path = next((ROOT / "data/ftw_polygons").glob(f"boundaries_{country}_*.parquet"))
        _POLYS[country] = gpd.read_parquet(path, columns=["geometry"]).to_crs(4326)
    return _POLYS[country]


def stack_nir(sr: Path, s2: Path, out: Path) -> None:
    """The user's stack_nir.py, called as its command line would be."""
    sys.path.insert(0, str(INFERENCE))
    import stack_nir as tool
    argv = sys.argv
    sys.argv = ["stack_nir.py", "--sr", str(sr), "--s2", str(s2), "--out", str(out)]
    try:
        tool.main()
    finally:
        sys.argv = argv


def labels(country: str, footprint, shape, transform) -> np.ndarray:
    polys = polygons(country)
    hit = polys.iloc[polys.sindex.query(footprint, predicate="intersects")].geometry
    # FTW polygons can be invalid (self-intersections in France): repair them before clipping
    clipped = [g for g in hit.make_valid().intersection(footprint) if not g.is_empty and g.area > 0]
    out = np.zeros(shape, np.uint8)
    if not clipped:
        return out
    rasterize(clipped, out=out, transform=transform, default_value=1)
    px = (abs(transform.a) + abs(transform.e)) / 2
    rings = [part.exterior.buffer(px) for g in clipped for part in getattr(g, "geoms", [g])
             if part.geom_type == "Polygon"]
    if rings:
        rasterize(rings, out=out, transform=transform, default_value=2)
    return out


def do_chip(job: dict) -> dict:
    country, aoi_id, out, sr = job["country"], job["aoi_id"], Path(job["out"]), Path(job["sr"])
    srs = {w: sr / country / w / f"{aoi_id}_SR3m.tif" for w in WINDOWS}
    if not all(p.exists() for p in srs.values()):
        return dict(country=country, aoi_id=aoi_id, status="no SR yet")
    paths = {}
    for w in WINDOWS:
        rel = f"{country}/{w}/{aoi_id}.tif"
        target = out / "planet" / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".part.tif")
            stack_nir(srs[w], ROOT / "data/ftw12" / country / "s2_images" / w / f"{aoi_id}.tif", tmp)
            tmp.replace(target)
        paths[w] = rel
    lab = f"{country}/labels/{aoi_id}.tif"
    target = out / "planet" / lab
    if not target.exists():
        with rasterio.open(out / "planet" / paths["window_a"]) as src:
            prof, shape, transform = src.profile, src.shape, src.transform
        footprint = gpd.GeoSeries.from_wkb([job["footprint"]]).iloc[0]
        prof.update(count=1, dtype="uint8", photometric="MINISBLACK", nodata=None)
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(target, "w", **prof) as dst:
            dst.write(labels(country, footprint, shape, transform)[None])
    return dict(country=country, aoi_id=aoi_id, status="ok", image_a_path=paths["window_a"],
                image_b_path=paths["window_b"], label_path=lab)


def panel(out: Path, row: dict, dst: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import binary_dilation
    with rasterio.open(out / "planet" / row["image_b_path"]) as f:
        img = f.read()
    with rasterio.open(out / "planet" / row["label_path"]) as f:
        lab = f.read(1)
    with rasterio.open(ROOT / "data/ftw" / row["country"] / "s2_images/window_b" / f"{row['aoi_id']}.tif") as f:
        s2 = f.read((1, 2, 3)).transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(s2, [1, 99])
    s2 = (np.clip((s2 - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    h, w = lab.shape
    s2 = np.asarray(Image.fromarray(s2).resize((w, h), Image.NEAREST))
    sr = img[:3].transpose(1, 2, 0).astype(np.uint8)
    over = sr.copy()
    over[binary_dilation(lab == 2)] = (255, 235, 0)
    cells = [("FTW Sentinel-2, window b", s2), ("SR (LDM 3 m), window b", sr), ("SR + label outlines", over),
             ("labels", CLASS_RGB[lab])]
    font = ImageFont.load_default(size=18)
    sheet = Image.new("RGB", (4 * (w + 6), h + 54), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 2), f"{row['country']} {row['aoi_id']} (rows as stored by FTW)", fill="black", font=font)
    for i, (title, cell) in enumerate(cells):
        draw.text((i * (w + 6) + 4, 26), title, fill="black", font=font)
        sheet.paste(Image.fromarray(cell), (i * (w + 6), 52))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sr", type=Path, required=True, help="folder with <country>/<window>/<aoi_id>_SR3m.tif")
    ap.add_argument("--name", required=True, help="layout folder name under data/ftp_layout/")
    ap.add_argument("--train", type=Path)
    ap.add_argument("--val", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--panels", type=int, default=0, help="panels per country")
    ap.add_argument("--countries", nargs="+", default=None, help="only these countries (per-chip files only)")
    ap.add_argument("--no-index", action="store_true",
                    help="write the per-chip files only; index.parquet and the split tables in a later full run")
    args = ap.parse_args()
    out = ROOT / "data/ftp_layout" / args.name
    sr = args.sr if args.sr.is_absolute() else ROOT / args.sr

    splits = []
    for split in ("train", "val", "test"):
        sel = getattr(args, split)
        if sel:
            splits.append(pd.read_csv(sel, sep=" ", names=["country", "aoi_id"], dtype=str).assign(split=split))
    splits = pd.concat(splits, ignore_index=True)
    if args.countries:
        splits = splits[splits.country.isin(args.countries)]
    jobs = []
    for country, g in splits.drop_duplicates(["country", "aoi_id"]).groupby("country"):
        chips = gpd.read_parquet(ROOT / f"data/ftw_chips/chips_{country}.parquet").to_crs(4326)
        chips["aoi_id"] = chips.aoi_id.astype(str)
        geom = dict(zip(chips.aoi_id, chips.geometry))
        jobs += [dict(country=country, aoi_id=a, out=str(out), sr=str(sr), footprint=geom[a].wkb) for a in g.aoi_id]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(do_chip, jobs, chunksize=8))
    res = pd.DataFrame(rows)
    ok = res[res.status == "ok"]
    print(f"{len(ok)} of {len(res)} chips written; {int((res.status != 'ok').sum())} without SR yet")
    if ok.empty or args.no_index:
        return
    geom = {(j["country"], j["aoi_id"]): gpd.GeoSeries.from_wkb([j["footprint"]]).iloc[0] for j in jobs}
    index = gpd.GeoDataFrame(ok.assign(patch_id=ok.aoi_id, usable_pair=True,
                                       geometry=[geom[(c, a)] for c, a in zip(ok.country, ok.aoi_id)])
                             .drop(columns=["aoi_id", "status"]), crs=4326)
    (out / "planet").mkdir(parents=True, exist_ok=True)
    index.to_parquet(out / "planet/index.parquet")
    have = set(zip(ok.country, ok.aoi_id))
    for country, g in splits.groupby("country"):
        g = g[[(country, a) in have for a in g.aoi_id]]
        if g.empty:
            continue
        path = out / "ftw" / country / f"chips_{country}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(dict(aoi_id=g.aoi_id.values, split=g.split.values),
                         geometry=[geom[(country, a)] for a in g.aoi_id], crs=4326).to_parquet(path)
    print(ok.merge(splits, on=["country", "aoi_id"]).groupby(["country", "split"]).size().unstack(fill_value=0)
          .to_string())
    if args.panels:
        for country, g in ok.groupby("country"):
            for row in g.head(args.panels).to_dict("records"):
                panel(out, row, ROOT / "results" / f"{args.name}_inputs" / country / f"{row['aoi_id']}.jpg")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
