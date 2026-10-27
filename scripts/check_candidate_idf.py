# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : scripts/check_candidate_idf.py
# ======================================================================================
"""Test a CANDIDATE IDF (unverified, supplied by user 24 Oct 2026) against the registry and AACRA Bole values (Tedla 2016, Table 4.2).
I = C * Tr**m / (d + b)**e   (d in minutes, I in mm/hr)
"""
import numpy as np
def I(Tr, d, C, m, b, e): return C * Tr**m / (d + b)**e
cand = (682.3, 0.21, 15.5, 0.82)       # candidate "ERA Region A2" constants: NOT VERIFIED
reg  = (520.4, 0.19, 14.5, 0.78)       # registry formula constants: NOT VERIFIED
tab  = {2: 42.5, 5: 54.8, 10: 68.0, 25: 81.4, 50: 94.2, 100: 108.5}
aacra = {2: {5: 86, 10: 73, 15: 63}, 100: {5: 138, 10: 119, 15: 105}}   # Tedla (2016), quoted in text

print("30-min intensity, mm/hr:  Tr | candidate | registry formula | registry table")
for Tr in tab: print(f"  {Tr:>3} | {I(Tr,30,*cand):6.1f} | {I(Tr,30,*reg):6.1f} | {tab[Tr]:6.1f}")

print("\nShort durations vs AACRA Bole (mm/hr): Tr, d | AACRA | candidate | ratio")
for Tr, row in aacra.items():
    for d, v in row.items():
        c = I(Tr, d, *cand); print(f"  {Tr:>3}, {d:>2} | {v:5.0f} | {c:6.1f} | {c/v:4.2f}")

print("\nReturn-period scaling 2 -> 100 yr (ratio of intensities):")
print("  candidate exponent m = 0.21 ->", round(50**0.21, 2))
for d in (5, 10, 15):
    r = aacra[100][d] / aacra[2][d]; print(f"  AACRA Bole at {d:>2} min -> {r:.2f}  (implied m = {np.log(r)/np.log(50):.3f})")
