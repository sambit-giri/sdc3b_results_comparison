"""
Pre-compute 1D and 2D marginals from team FITS posteriors and write posteriors_data.js.
Run once before opening compare_posterior.html.

Usage:
    python posterior_for_html.py
"""

import json
import numpy as np
from glob import glob
from astropy.io import fits

MOCKS_DIR = "mocks"
OUTPUT    = "posteriors_data.js"
N         = 100          # grid resolution (must match FITS)
COORDS    = np.linspace(0, 1, N)

PARAM_NAMES  = ["x_HI,1", "x_HI,2", "x_HI,3"]
PARAM_LABELS = ["x_{HI,1}", "x_{HI,2}", "x_{HI,3}"]

TRUE_VALUES = {
    "PS1": [0.69960, 0.40125, 0.12525],
}


def marginal_1d(grid, axis):
    axes = [i for i in range(3) if i != axis]
    return grid.sum(axis=tuple(axes))


def marginal_2d(grid, ax_i, ax_j):
    sum_axis = tuple(k for k in range(3) if k not in (ax_i, ax_j))
    m = grid.sum(axis=sum_axis)
    if ax_i > ax_j:
        m = m.T
    return m


def normalise(arr):
    s = arr.sum()
    return ((arr / s) if s > 0 else arr).flatten().tolist()


def credible_thresholds(m2d, fracs=(0.68, 0.95)):
    """Return probability density thresholds enclosing the given credible fractions."""
    flat = np.sort(m2d.flatten())[::-1]   # descending
    cumsum = np.cumsum(flat)
    return [float(flat[min(np.searchsorted(cumsum, f), len(flat)-1)]) for f in fracs]


def process_team(team_name, ps_name):
    path = f"{MOCKS_DIR}/{team_name}/{team_name}_{ps_name}.fits"
    try:
        grid = fits.open(path)[0].data.astype(np.float64)
    except FileNotFoundError:
        return None

    grid /= grid.sum()

    marg1d = [normalise(marginal_1d(grid, i)) for i in range(3)]

    # 2D marginals with 68% and 95% credible interval thresholds
    marg2d = {}
    levels = {}
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        m = marginal_2d(grid, i, j)
        key = f"{i}_{j}"
        marg2d[key] = normalise(m)
        levels[key] = credible_thresholds(m)   # [thresh_68, thresh_95]

    return {"marg1d": marg1d, "marg2d": marg2d, "levels": levels}


def main():
    ps_list = ["PS1"]
    teams = sorted(
        [f"Team_{i}" for i in range(1, 27)],
        key=lambda t: int(t.split("_")[1]),
    )

    output = {
        "coords": COORDS.tolist(),
        "param_labels": PARAM_LABELS,
        "param_names": PARAM_NAMES,
        "true_values": TRUE_VALUES,
        "teams": teams,
        "ps_list": ps_list,
        "data": {},
    }

    for ps in ps_list:
        output["data"][ps] = {}
        for team in teams:
            print(f"  Processing {team} {ps} ...", end="\r")
            result = process_team(team, ps)
            if result is not None:
                output["data"][ps][team] = result
            else:
                print(f"\n  [SKIP] {team}/{ps} — file not found")

    with open(OUTPUT, "w") as f:
        f.write("const POSTERIORS_DATA=")
        json.dump(output, f, separators=(",", ":"))
        f.write(";")

    size_mb = __import__("os").path.getsize(OUTPUT) / 1e6
    print(f"\nWrote {OUTPUT}  ({size_mb:.1f} MB, {len(teams)} teams)")


if __name__ == "__main__":
    main()
