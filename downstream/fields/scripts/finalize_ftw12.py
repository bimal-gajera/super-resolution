#!/usr/bin/env python3
"""After scripts/ftw12_chips.py and scripts/ftw12_repair.py: which 12-band chips are FTW's exact chips, and the final
chip lists.

A chip-date is exact if its 12-band file reproduces the published FTW chip (first pass `equal_share` >= 0.99, or
identified in the repair pass). A chip is kept only if both dates are exact; the files of the other chips are moved
to data/ftw12_inexact/ so that folder-wise SR runs skip them.

    python scripts/finalize_ftw12.py

Writes data/ftw12/scenes_final.csv (country, aoi_id, window, scene, exact), data/selections/<list>_exact.txt for
ftp_train / ftp_val / ftp_test / ftp_test_dense10, and data/selections/exact_chips_per_country.csv.
Running it again changes nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from ftw12_chips import MATCH

ROOT = Path(__file__).resolve().parent.parent
LISTS = ["ftp_train", "ftp_val", "ftp_test", "ftp_test_dense10"]


def read(folder: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype={"aoi_id": str}) for p in sorted(folder.glob("*.csv"))]
    return pd.concat(frames).drop_duplicates(["country", "aoi_id", "window"], keep="last") if frames else None


def selection(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"data/selections/{name}.txt", sep=" ", names=["country", "aoi_id"], dtype=str)


def main() -> None:
    first = read(ROOT / "data/ftw12/scenes")[["country", "aoi_id", "window", "scene_id", "equal_share"]]
    rep = read(ROOT / "data/ftw12/scenes_repair")
    first = first.rename(columns={"scene_id": "scene_first"})
    if rep is not None:
        first = first.merge(rep[["country", "aoi_id", "window", "search", "scene_id"]],
                            on=["country", "aoi_id", "window"], how="left")
    else:
        first = first.assign(search=None, scene_id=None)
    first["exact"] = (first.equal_share >= MATCH) | (first.search == "identified")
    first["scene"] = first.scene_id.where(first.search == "identified", first.scene_first)
    wanted = selection("ftp_all")
    grid = pd.DataFrame([(c, a, w) for c, a in zip(wanted.country, wanted.aoi_id) for w in ("window_a", "window_b")],
                        columns=["country", "aoi_id", "window"])
    final = grid.merge(first[["country", "aoi_id", "window", "scene", "exact"]], how="left",
                       on=["country", "aoi_id", "window"])
    final["exact"] = final.exact.fillna(False).astype(bool)
    final.to_csv(ROOT / "data/ftw12/scenes_final.csv", index=False)
    print(f"exact chip-dates: {int(final.exact.sum())} of {len(final)}")

    bad = set(zip(final.country[~final.exact], final.aoi_id[~final.exact]))
    moved = 0
    for c, a in bad:
        for w in ("window_a", "window_b"):
            src = ROOT / f"data/ftw12/{c}/s2_images/{w}/{a}.tif"
            if src.exists():
                dst = ROOT / f"data/ftw12_inexact/{c}/s2_images/{w}/{a}.tif"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)
                moved += 1
    print(f"chips left out: {len(bad)} (files moved now: {moved})")
    kept = []
    for name in LISTS:
        d = selection(name)
        d = d[[(c, a) not in bad for c, a in zip(d.country, d.aoi_id)]]
        (ROOT / f"data/selections/{name}_exact.txt").write_text("".join(f"{c} {a}\n" for c, a in zip(d.country, d.aoi_id)))
        print(f"{name}_exact: {len(d)} chips")
        if name != "ftp_test_dense10":
            kept.append(d)
    counts = pd.concat(kept).drop_duplicates().country.value_counts()
    counts.to_csv(ROOT / "data/selections/exact_chips_per_country.csv")


if __name__ == "__main__":
    main()
