                                                                   
                                         

                                     

                                                                        
                                                                       
                                                        
                                                                        
                                                                       
                                                            
                                                                      
                                                                      
               

                                                              
                                                          
   
from __future__ import annotations

import glob
import itertools
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cpcb_index import index_to_concentration, is_censored

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW10 = os.path.join(ROOT, "data", "raw", "bengaluru_10station_cpcb_subindex")
RAW4 = os.path.join(ROOT, "data", "raw", "bengaluru_4station_cpcb_raw")

POLLUTANTS = ["PM2.5", "PM10", "NO2", "O3"]

SHORT = {
    "BTM Layout, Bengaluru - CPCB": "BTM",
    "BWSSB Kadabesanahalli, Bengaluru - CPCB": "BWSSB",
    "Bapuji Nagar, Bengaluru - KSPCB": "Bapuji",
    "City Railway Station, Bengaluru - KSPCB": "CityRly",
    "Hebbal, Bengaluru - KSPCB": "Hebbal",
    "Hombegowda Nagar, Bengaluru - KSPCB": "Hombegowda",
    "Jayanagar 5th Block, Bengaluru - KSPCB": "Jayanagar",
    "Peenya, Bengaluru - CPCB": "Peenya",
    "Sanegurava Halli, Bengaluru - KSPCB": "Sanegurava",
    "Silk Board, Bengaluru - KSPCB": "SilkBoard",
}
RAW4_FILES = {"btm": "BTM", "jayanagar": "Jayanagar", "peenya": "Peenya", "silkboard": "SilkBoard"}


def station_coordinates() -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(RAW10, "air_data_stations.csv"))
    meta = meta[meta["name"].isin(SHORT)].copy()
    meta["station"] = meta["name"].map(SHORT)
    return meta.set_index("station")[["latitude", "longitude", "name"]]


def load_b10(pollutants=POLLUTANTS) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW10, "20*_air_pollution_data_*.csv.gz"))):
        d = pd.read_csv(f)
        d = d[d["name"].isin(SHORT)]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["station"] = d["name"].map(SHORT)
    t = pd.to_datetime(d["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    other = t.isna()
    if other.any():                                                   
        t[other] = pd.to_datetime(d.loc[other, "datetime"], format="%A, %d %b %Y, %I:%M %p", errors="coerce")
    d = d[t.notna()].copy()
    d["t"] = t[t.notna()] - pd.Timedelta(hours=1)
    d = d.rename(columns={"OZONE": "O3"})
    cols = {}
    for p in pollutants:
        idx = d[p].to_numpy(dtype=float)
        conc = index_to_concentration(p, idx)
        conc[is_censored(idx)] = np.nan                                 
        cols[p] = conc
    long = pd.DataFrame({"t": d["t"], "station": d["station"], **cols})
    long = long.groupby(["t", "station"]).mean(numeric_only=True)                            
    wide = long.unstack("station")
    wide.columns = wide.columns.swaplevel(0, 1)
    wide = wide.sort_index(axis=1)
    wide.columns.names = ["station", "pollutant"]
    return wide.sort_index()


def load_b4(pollutants=POLLUTANTS) -> pd.DataFrame:
    rename = {"PM2.5 (µg/m³)": "PM2.5", "PM10 (µg/m³)": "PM10", "NO2 (µg/m³)": "NO2", "Ozone (µg/m³)": "O3"}
    per_station = {}
    for f in sorted(glob.glob(os.path.join(RAW4, "*.csv"))):
        base = os.path.basename(f).lower()
        key = next((k for k in RAW4_FILES if base.startswith(k) or k.replace("board", "_board") in base), None)
        if key is None:
            continue
        d = pd.read_csv(f, na_values=["NA", "None", ""])
        d = d.rename(columns=rename)
        d["t"] = pd.to_datetime(d["Timestamp"], format="%Y-%m-%d %H:%M:%S")
        d = d.set_index("t")[[p for p in pollutants if p in d.columns]]
        per_station.setdefault(RAW4_FILES[key], []).append(d)
    frames = []
    for s, lst in per_station.items():
        d = pd.concat(lst).sort_index()
        d = d[~d.index.duplicated(keep="first")]
        d.columns = pd.MultiIndex.from_product([[s], d.columns], names=["station", "pollutant"])
        frames.append(d)
    wide = pd.concat(frames, axis=1).sort_index(axis=1)
    return wide.sort_index()


def load(dataset: str) -> pd.DataFrame:
    if dataset == "b10":
        return load_b10()
    if dataset == "b4":
        return load_b4()
    raise ValueError(dataset)


                                                                             
                          
                                                                             

@dataclass
class Selection:
    stations: list
    pollutants: list
    n_complete_hours: int
    start: pd.Timestamp
    end: pd.Timestamp
    table: pd.DataFrame = field(repr=False)


def coverage_table(wide: pd.DataFrame) -> pd.DataFrame:
                                                                      
                                 
    rows = []
    for s in wide.columns.get_level_values(0).unique():
        block = wide[s]
        active = block.notna().any(axis=1)
        if active.sum() == 0:
            continue
        span = block.loc[active.idxmax(): active[::-1].idxmax()]
        for p in block.columns:
            rows.append({"station": s, "pollutant": p, "coverage": span[p].notna().mean(),
                         "n_hours": int(span[p].notna().sum())})
    return pd.DataFrame(rows).pivot(index="station", columns="pollutant", values="coverage")


def select_subset(wide: pd.DataFrame, min_station_coverage=0.5, min_complete_hours=3000,
                  pollutant_candidates=POLLUTANTS) -> Selection:
                                                                      

                                                                          
                                                                           
                                                                          
                                                                    
                                                     
       
    cov = coverage_table(wide)
    cands = []
    for k in range(len(pollutant_candidates), 1, -1):
        for P in itertools.combinations(pollutant_candidates, k):
            P = list(P)
            ok = cov[P].ge(min_station_coverage).all(axis=1)
            S = list(cov.index[ok])
            if len(S) < 3:
                continue
            cols = [(s, p) for s in S for p in P]
            complete = wide[cols].notna().all(axis=1)
            n = int(complete.sum())
            cands.append({"pollutants": P, "stations": S, "n_stations": len(S), "n_pollutants": len(P),
                          "cells": len(S) * len(P), "n_complete_hours": n,
                          "start": wide.index[complete].min() if n else pd.NaT,
                          "end": wide.index[complete].max() if n else pd.NaT})
    table = pd.DataFrame(cands).sort_values(["cells", "n_complete_hours"], ascending=False)
    ok = table[table["n_complete_hours"] >= min_complete_hours]
    best = ok.iloc[0] if len(ok) else table.iloc[0]
    return Selection(list(best["stations"]), list(best["pollutants"]), int(best["n_complete_hours"]),
                     best["start"], best["end"], table)


                                                                             
                            
                                                                             

@dataclass
class Dataset:
    name: str
    stations: list
    pollutants: list
    coords: pd.DataFrame                                 
    X: np.ndarray                                                                           
    times: pd.DatetimeIndex
    split: np.ndarray                                                
    mean: np.ndarray = None                                    
    std: np.ndarray = None

    @property
    def n(self):
        return len(self.stations)

    @property
    def p(self):
        return len(self.pollutants)

    def standardise(self):
        tr = self.X[self.split == 0]
        self.mean = tr.mean(axis=0)
        self.std = tr.std(axis=0, ddof=1)
        return self

    def Z(self, part=None):
                                                                       
        Z = (self.X - self.mean) / self.std
        return Z if part is None else Z[self.split == part]

    def unstandardise(self, Zsub, station_idx):
        return Zsub * self.std[station_idx] + self.mean[station_idx]

    def rows(self, part):
        return np.where(self.split == part)[0]


def build_dataset(name: str, wide: pd.DataFrame, sel: Selection, fractions=(0.6, 0.2, 0.2)) -> Dataset:
    cols = [(s, p) for s in sel.stations for p in sel.pollutants]
    sub = wide[cols].dropna(how="any")
    T = len(sub)
    X = sub.to_numpy(dtype=float).reshape(T, len(sel.stations), len(sel.pollutants))
    n_tr = int(round(fractions[0] * T))
    n_va = int(round(fractions[1] * T))
    split = np.zeros(T, dtype=int)
    split[n_tr:n_tr + n_va] = 1
    split[n_tr + n_va:] = 2
    coords = station_coordinates().loc[sel.stations, ["latitude", "longitude"]]
    ds = Dataset(name, list(sel.stations), list(sel.pollutants), coords, X, sub.index, split)
    return ds.standardise()


def haversine_km(coords: pd.DataFrame) -> np.ndarray:
    lat = np.radians(coords["latitude"].to_numpy())
    lon = np.radians(coords["longitude"].to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
