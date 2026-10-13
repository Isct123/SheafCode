                                                                          
                                                                        
                                                                         
                                                                          
from __future__ import annotations

import glob
import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cpcb_index import index_to_concentration, is_censored

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_INDIA = os.path.join(ROOT, "data", "raw", "india_cpcb_subindex")
POLLUTANTS = ["PM2.5", "PM10", "NO2", "O3"]

REGIONS = {
    "delhi": ["Delhi"],
    "ncr": ["Delhi", "Noida", "Greater Noida", "Gurugram", "Ghaziabad", "Faridabad"],
    "mumbai": ["Mumbai", "Navi Mumbai"],
    "kolkata": ["Kolkata", "Howrah"],
    "hyderabad": ["Hyderabad"],
    "bengaluru": ["Bengaluru"],
    "india": None,
    "delhi2025": "raw",
    "bengaluru_raw": "raw",
}


def load_india_long() -> pd.DataFrame:
    cache = os.path.join(ROOT, "data", "processed", "india_long.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW_INDIA, "20*_air_pollution_data_india.csv.gz"))):
        frames.append(pd.read_csv(f))
    d = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(d["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    other = t.isna()
    if other.any():
        t[other] = pd.to_datetime(d.loc[other, "datetime"], format="%A, %d %b %Y, %I:%M %p", errors="coerce")
    d = d[t.notna()].copy()
    d["t"] = t[t.notna()] - pd.Timedelta(hours=1)
    d = d.rename(columns={"OZONE": "O3"})
    for p in POLLUTANTS:
        idx = d[p].to_numpy(dtype=float)
        conc = index_to_concentration(p, idx)
        conc[is_censored(idx)] = np.nan
        d[p] = conc
    d = d[["t", "id", "name", "cityid", "stateid", "latitude", "longitude"] + POLLUTANTS]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    d.to_parquet(cache)
    return d


def load_delhi2025_wide(pollutants=POLLUTANTS):
                                                                
                                                                      
                                                        
    d = pd.read_parquet(os.path.join(ROOT, "data", "raw", "delhi_2025_openaq_cpcb", "cpcb_hourly.parquet"))
    d = d.rename(columns={"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO2", "o3": "O3"})
    d["t"] = pd.to_datetime(d["datetime"])
    d["id"] = d["station"]
    meta = d.groupby("id").agg(name=("station", "first"), city=("city", "first"), latitude=("lat", "first"), longitude=("lon", "first"))
    long = d.groupby(["t", "id"])[list(pollutants)].mean()
    wide = long.unstack("id"); wide.columns = wide.columns.swaplevel(0, 1); wide = wide.sort_index(axis=1)
    wide.columns.names = ["station", "pollutant"]
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(full).sort_index(), meta


def load_bengaluru_raw_wide(pollutants=POLLUTANTS):
                                                                        
                                                                             
    from .data import load_b4, station_coordinates
    wide = load_b4(pollutants)
    coords = station_coordinates()
    meta = pd.DataFrame({"name": [coords.loc[s, "name"] for s in wide.columns.get_level_values(0).unique()],
                         "city": "Bengaluru",
                         "latitude": [coords.loc[s, "latitude"] for s in wide.columns.get_level_values(0).unique()],
                         "longitude": [coords.loc[s, "longitude"] for s in wide.columns.get_level_values(0).unique()]},
                        index=wide.columns.get_level_values(0).unique())
    return wide, meta


def region_wide(region: str, pollutants=POLLUTANTS):
    if region == "delhi2025":
        return load_delhi2025_wide(pollutants)
    if region == "bengaluru_raw":
        return load_bengaluru_raw_wide(pollutants)
    d = load_india_long()
    cities = REGIONS[region]
    if cities is not None:
        d = d[d["cityid"].isin(cities)]
    meta = d.groupby("id").agg(name=("name", "first"), city=("cityid", "first"), latitude=("latitude", "first"),
                               longitude=("longitude", "first"))
    long = d.groupby(["t", "id"])[list(pollutants)].mean()
    wide = long.unstack("id")
    wide.columns = wide.columns.swaplevel(0, 1)
    wide = wide.sort_index(axis=1)
    wide.columns.names = ["station", "pollutant"]
    return wide.sort_index(), meta


@dataclass
class MaskedDataset:
    name: str
    stations: list
    pollutants: list
    coords: pd.DataFrame
    meta: pd.DataFrame
    X: np.ndarray                                              
    times: pd.DatetimeIndex
    split: np.ndarray
    transform: str = "none"                   
    mean: np.ndarray = None
    std: np.ndarray = None

    n = property(lambda self: len(self.stations))
    p = property(lambda self: len(self.pollutants))

    def set_transform(self, transform):
                                                                                    
        self.transform = transform
        tr = self._fwd(self.X[self.split == 0])
        self.mean = np.nanmean(tr, axis=0)
        self.std = np.nanstd(tr, axis=0, ddof=1)
        self.std[~np.isfinite(self.std) | (self.std == 0)] = 1.0
        self.hod = None
        if transform == "diurnal":
            h = self.times[self.split == 0].hour.to_numpy()
            self.hod = np.zeros((24,) + self.mean.shape)
            for k in range(24):
                self.hod[k] = np.nanmean(tr[h == k], axis=0) - self.mean
            self.hod = np.nan_to_num(self.hod)
        return self

    def _fwd(self, X):
        return np.log1p(np.clip(X, 0, None)) if self.transform in ("log", "diurnal") else X

    def _inv(self, Y):
        return np.expm1(Y) if self.transform in ("log", "diurnal") else Y

    def Z(self, part=None):
        Y = self._fwd(self.X) - self.mean
        if self.hod is not None:
            Y = Y - self.hod[self.times.hour.to_numpy()]
        Z = Y / self.std
        return Z if part is None else Z[self.split == part]

    def raw_from_Z(self, Zsub, rows, station_idx):
                                                                                
        Y = Zsub * self.std[station_idx] + self.mean[station_idx]
        if self.hod is not None:
            Y = Y + self.hod[self.times[rows].hour.to_numpy()][:, station_idx, :]
        return self._inv(Y)

    def to_raw(self, Zsub, station_idx):
        return self._inv(Zsub * self.std[station_idx] + self.mean[station_idx])

    def rows(self, part):
        return np.where(self.split == part)[0]


def select_region(region: str, min_station_coverage=0.5, min_row_fraction=0.7, min_hours=3000,
                  pollutant_candidates=POLLUTANTS):
                                                                             
                                                                       
                                                                            
                                                                      
    wide, meta = region_wide(region)
                                           
    cov = {}
    for s in wide.columns.get_level_values(0).unique():
        blk = wide[s]
        act = blk.notna().any(axis=1)
        if act.sum() < 24 * 60:
            continue
        span = blk.loc[act.idxmax(): act[::-1].idxmax()]
        cov[s] = span.notna().mean()
    cov = pd.DataFrame(cov).T
    best = None
    for k in range(len(pollutant_candidates), 1, -1):
        for P in itertools.combinations(pollutant_candidates, k):
            P = list(P)
            S = list(cov.index[cov[P].ge(min_station_coverage).all(axis=1)])
            if len(S) < 4:
                continue
            cols = [(s, p) for s in S for p in P]
            frac = wide[cols].notna().mean(axis=1)
            keep = frac >= min_row_fraction
            n = int(keep.sum())
            score = (len(S) * len(P), n)
            if n >= min_hours and (best is None or score > best[0]):
                best = (score, S, P, keep)
    if best is None:
        raise RuntimeError("no configuration meets the coverage rule")
    _, S, P, keep = best
    cols = [(s, p) for s in S for p in P]
    sub = wide.loc[keep, cols]
    T = len(sub)
    X = sub.to_numpy(dtype=float).reshape(T, len(S), len(P))
    n_tr, n_va = int(round(0.6 * T)), int(round(0.2 * T))
    split = np.zeros(T, int); split[n_tr:n_tr + n_va] = 1; split[n_tr + n_va:] = 2
    m = meta.loc[S]
    coords = m[["latitude", "longitude"]]
    ds = MaskedDataset(region, list(S), list(P), coords, m, X, sub.index, split)
    return ds.set_transform("none")


                                                                              
                                                                        
                                                                             
                                                                              

def complete_hourly(wide):
                                                                     
                                                                      
                                                                            
    full = pd.date_range(wide.index.min().floor("h"), wide.index.max().floor("h"), freq="h")
    out = wide.reindex(full)
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert (np.diff(out.index.values).astype("timedelta64[h]").astype(int) == 1).all()
    return out


def select_region_v4(region: str, min_station_coverage=0.5, min_good_hours=2000, good_row_fraction=0.7,
                     pollutant_candidates=POLLUTANTS, split=(0.6, 0.2, 0.2)):
                                                           

                                                                      
                                                                       
                                         
                                                                        
                                                               
                                                                           
                                                                       
                                                               
                                                                       
                            
                                                                   
                                                           
       
    wide, meta = region_wide(region)
    wide = complete_hourly(wide)
                                                                                    
                                                                                       
                                                            
    starts = sorted({wide[s].notna().any(axis=1).idxmax() for s in wide.columns.get_level_values(0).unique()})
    best = None
    for start in starts:
        w = wide[wide.index >= start]
        T = len(w)
        if T < 24 * 240:
            continue
        n_tr = int(round(split[0] * T))
        train = w.iloc[:n_tr]
        cov = {}
        for s in w.columns.get_level_values(0).unique():
            blk = train[s]
            act = blk.notna().any(axis=1)
            if act.sum() < 24 * 30:
                continue
            span = blk.loc[act.idxmax():]
            cov[s] = span.notna().mean()
        if not cov:
            continue
        cov = pd.DataFrame(cov).T
        for k in range(len(pollutant_candidates), 1, -1):
            for P in itertools.combinations(pollutant_candidates, k):
                P = list(P)
                if not all(p in cov.columns for p in P):
                    continue
                S = list(cov.index[cov[P].ge(min_station_coverage).all(axis=1)])
                if len(S) < 4:
                    continue
                cols = [(s, p) for s in S for p in P if (s, p) in train.columns]
                if len(cols) < len(S) * len(P):
                    continue
                frac = train[cols].notna().mean(axis=1)
                good = int((frac >= good_row_fraction).sum())
                score = (len(S) * len(P), good)
                if good >= min_good_hours and (best is None or score > best[0]):
                    best = (score, S, P, start, w.index[n_tr - 1], cov.loc[S, P].round(3).to_dict())
    if best is None:
        raise RuntimeError("no configuration meets the training-only coverage rule")
    _, S, P, start, cov_end, cov_table = best
    wide = wide[wide.index >= start]
    T = len(wide); n_tr = int(round(split[0] * T))
    cols = [(s, p) for s in S for p in P]
    sub = wide[cols]
    first = sub.notna().any(axis=1).idxmax()
    sub = sub[sub.index >= first]
                                                                                       
                                                                                      
    T2 = len(sub); n_tr2, n_va2 = int(round(split[0] * T2)), int(round(split[1] * T2))
    split_arr = np.zeros(T2, int); split_arr[n_tr2:n_tr2 + n_va2] = 1; split_arr[n_tr2 + n_va2:] = 2
    assert sub.index[n_tr2 + n_va2] >= wide.index[n_tr - 1], "coverage window overlaps the test block"
    X = sub.to_numpy(dtype=float).reshape(len(sub), len(S), len(P))
    m = meta.loc[S]
    ds = MaskedDataset(region, list(S), list(P), m[["latitude", "longitude"]], m, X, sub.index, split_arr)
    ds.selection = dict(rule="v4 training-only coverage", min_station_coverage=min_station_coverage, min_good_hours=min_good_hours,
                        good_row_fraction=good_row_fraction, split=list(split), coverage_window_end=str(cov_end),
                        train_end=str(sub.index[n_tr2 - 1]), val_end=str(sub.index[n_tr2 + n_va2 - 1]),
                        timeline_hours=int(len(sub)), timeline_start=str(start), coverage_train=cov_table)
    return ds.set_transform("none")

