                                                                     

                                                                  
                                                                   
                                                                   
                                                                       
                                                                      
                                                                      
                                                                        
                                                                    
                                                                       
              

                                                            
   
from __future__ import annotations

import numpy as np

                                                        
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
              (90, 120, 200, 300), (120, 250, 300, 400), (250, 380, 400, 500)],
    "PM10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
             (250, 350, 200, 300), (350, 430, 300, 400), (430, 510, 400, 500)],
    "NO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200),
            (180, 280, 200, 300), (280, 400, 300, 400), (400, 520, 400, 500)],
    "O3": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200),
           (168, 208, 200, 300), (208, 748, 300, 400), (748, 1288, 400, 500)],
    "SO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200),
            (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2400, 400, 500)],
    "NH3": [(0, 200, 0, 50), (200, 400, 50, 100), (400, 800, 100, 200),
            (800, 1200, 200, 300), (1200, 1800, 300, 400), (1800, 2400, 400, 500)],
    "CO": [(0, 1.0, 0, 50), (1.0, 2.0, 50, 100), (2.0, 10, 100, 200),
           (10, 17, 200, 300), (17, 34, 300, 400), (34, 51, 400, 500)],
}


def concentration_to_index(pollutant: str, c):
                                                             
    c = np.asarray(c, dtype=float)
    out = np.full(c.shape, np.nan)
    for lo, hi, ilo, ihi in BREAKPOINTS[pollutant]:
        m = (c >= lo) & (c <= hi) & np.isnan(out)
        out[m] = ilo + (c[m] - lo) * (ihi - ilo) / (hi - lo)
    top = BREAKPOINTS[pollutant][-1]
    out[(c > top[1]) & np.isnan(out)] = 500.0
    return out


def index_to_concentration(pollutant: str, idx):
                                                                         

                                                                   
                                                            
       
    idx = np.asarray(idx, dtype=float)
    out = np.full(idx.shape, np.nan)
    for lo, hi, ilo, ihi in BREAKPOINTS[pollutant]:
        m = (idx >= ilo) & (idx <= ihi) & np.isnan(out)
        out[m] = lo + (idx[m] - ilo) * (hi - lo) / (ihi - ilo)
    return out


def is_censored(idx):
    idx = np.asarray(idx, dtype=float)
    return idx >= 500.0
