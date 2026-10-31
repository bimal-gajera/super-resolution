#!/usr/bin/env python3
"""The Fields of the Planet paper's chip sets as chip lists (`<country> <aoi_id>` per line).

The paper trains and tests on FTW chips that have a usable PlanetScope pair (`usable_pair` in FTP's index.parquet),
with FTW's own splits (chips_<country>.parquet): train = all 25 regions, val = Austria, test = all regions (the
headline numbers average the ten dense-label countries).

    python scripts/make_chip_lists.py

Reads data/ftp_index/index.parquet and data/ftw_chips/chips_*.parquet; writes data/selections/ftp_train.txt,
ftp_val.txt, ftp_test.txt, ftp_test_dense10.txt and ftp_all.txt (their union).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DENSE10 = ["belgium", "cambodia", "croatia", "germany", "latvia", "lithuania", "portugal", "slovenia",
           "south_africa", "sweden"]                       # ftw_planet.evaluation.DENSE_LABEL_COUNTRIES


def write(path: Path, d: pd.DataFrame) -> None:
    path.write_text("".join(f"{c} {a}\n" for c, a in zip(d.country, d.patch_id)))
    print(f"{path.name}: {len(d)} chips")


def main() -> None:
    idx = pq.ParquetFile(ROOT / "data/ftp_index/index.parquet").read(columns=["patch_id", "country", "usable_pair"])
    idx = idx.to_pandas()
    idx = idx[idx.usable_pair == True]                                  # noqa: E712
    idx["patch_id"] = idx.patch_id.astype(str)
    rows = []
    for p in sorted((ROOT / "data/ftw_chips").glob("chips_*.parquet")):
        ch = gpd.read_parquet(p)
        rows.append(pd.DataFrame(dict(patch_id=ch.aoi_id.astype(str), split=ch.split, country=p.stem[len("chips_"):])))
    m = pd.concat(rows).merge(idx[["patch_id", "country"]], on=["country", "patch_id"])
    out = ROOT / "data/selections"
    out.mkdir(parents=True, exist_ok=True)
    train, test = m[m.split == "train"], m[m.split == "test"]
    val = m[(m.split == "val") & (m.country == "austria")]
    write(out / "ftp_train.txt", train)
    write(out / "ftp_val.txt", val)
    write(out / "ftp_test.txt", test)
    write(out / "ftp_test_dense10.txt", test[test.country.isin(DENSE10)])
    write(out / "ftp_all.txt", pd.concat([train, val, test]).drop_duplicates(["country", "patch_id"]))


if __name__ == "__main__":
    main()
