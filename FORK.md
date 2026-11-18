# plumeIR fork for the Reykjanes OP-FTIR retrievals

Fork of plumeIR by J.-F. Smekens (MIT licence,
[jfsmekens/plumeIR](https://github.com/jfsmekens/plumeIR)), used to retrieve
gas ratios from OP-FTIR spectra of the 2021-2025 Reykjanes Peninsula
eruptions with the two-layer protocol of Smekens et al. (2024, JGR
Atmospheres 129, e2023JD040574).

## Base

Branch `reykjanes-fixes` is plumeIR's GitHub `main` (commit 1e27bb3,
August 2026) with the commits below on top. The release archived on Zenodo
with the paper (doi:10.5281/zenodo.10846166) is an older state of the code
that treats the layer temperatures differently; it is not used here.

## Commits (`git log main..reykjanes-fixes`)

Each commit message gives the reason for every change and, where measured,
its size.

| # | commit | retrieved columns |
|---|---|---|
| 1 | Windows paths, newer Midac files, fit start built after `force_param` | change only where the old code failed (a run after one failed fit) |
| 2 | Instrument line shape: FOV term at the window centre, continuous in the FOV, exact OPD sampling | change wherever the FOV matters |
| 3 | Reference cross-sections: ambient layer at its own temperature and pressure, RFM run checked, line list chosen at run time | change through the ambient temperature and pressure |
| 4 | Forward model: a gas in both layers, emission optical depth of continuum gases, saturation diagnostic | change only for plume-layer H2O with `model_gas_emission` |
| 5 | One cross-section for a layer temperature that is held, a calibration only when it varies | the cross-section is exact instead of the degree-10 polynomial's value; 1 RFM run per gas instead of 22 |
| 6 | `headless.py` batch runner, requirements | none (new file) |
| 7 | Tests | none |
| 8 | This file | none |
| 9 | Reference cross-sections from several RFM runs at a time | none (bit-identical); the reference step takes about 1/N of the time on N cores |
| 10 | A continuum gas (H2O) follows a fitted or imposed temperature of its layer | change only where the temperature of a layer with H2O varies; it was ignored |
| 11 | `headless.py`: ambient columns held fixed and lower bounds of parameters set from the job | none unless a job sets `fix_atm_columns` or `param_min` |
| 12 | Instrument line shape of plumeIR v3 (`ils.py`, ported from its v3-preview branch): exact pedestal, far tails as a broad mean, FOV broadening and shift exact on the interferogram, kernel length fixed for a fit, kernel centred | change wherever lines are resolved; plumeIR v3 measured layer columns 2.4-4.7 % low from the truncation alone; on 2024-02-08 Southern fissure 1 columns 1-3 % higher than with commit 2, the set's ratios within 1 sigma (CO2/SO2 -1.9 %, CO2/HCl -4.4 %, SO2/HCl -2.4 %) |
| 13 | This file: the maximum OPD of the Reykjanes MIDAC files | none |

## Cross-sections between temperature nodes (tested 18 November 2026)

plumeIR v3 replaces the degree-10 polynomial in temperature (21 RFM runs, plume 200-800 K) by linear
interpolation between RFM nodes. Against RFM run at the temperature itself, between nodes (295-565 K, 1000 hPa,
1 km), the polynomial is within 0.001-0.011 % at the strongest lines of SO2, CO2 and CO, linear interpolation on
the same nodes 0.03-0.17 %; the H2O amount polynomial (scalings 0-4) is within 0.003 %, linear 0.02-1.2 %. The
fork keeps the polynomial.

## Maximum OPD of the MIDAC files (tested 18 November 2026)

plumeIR v3 derives `max_opd` from the file as POINTS / (2 LWN), 1.037 cm for 32768 points at 15798 cm-1, which
holds for a double-sided interferogram. The Reykjanes MIDAC files (Essential FTIR 3.20) record a single-sided
forward interferogram of 32768 points with ZPD at point 1450, Mertz phase correction, triangle apodisation, zero
fill 2x and one point per laser fringe, so the maximum OPD is (32768 - 1450) / 15798.04 = 1.98 cm. With `ils.py`
and `max_opd` fitted together with the FOV, bright spectra still prefer a wider line core (an effective
`max_opd` of 0.5-0.9 cm, falling with wavenumber, with the FOV at 0.06-0.07 rad): the instrument broadening
scales with wavenumber, as a FOV does, but is smoother than the uniform-FOV box. The details are in the
workflow's docs/PLUMEIR.md (Maximum OPD).

## What is new to set

| where | name | default | meaning |
|---|---|---|---|
| `*RETRIEVAL` | `nu_shift_max` | 1.0 | bound of the wavenumber shift [cm-1] |
| `*RETRIEVAL` | `fov_min`, `fov_max` | 0.01, 0.15 | bounds of a fitted FOV [rad]; `inf` for no upper bound |
| environment | `PLUMEIR_HIT` | `./RFM/Hitran/hitran_2020.bin` | RFM line list, an RFM `.bin` or a HITRAN `.par` |
| environment | `PLUMEIR_HIT_SHA256` | computed | sha256 of that file, so it is not read again |
| environment | `PLUMEIR_RFM_THREADS` | number of CPUs | RFM runs at a time while the reference cross-sections are made |
| config | `<gas>_plume` | | name of the plume-layer parameter of a gas that is also an ambient gas |
| `headless.py` job | `fix_atm_columns` | | {gas: column}: ambient gases held at that column [molec cm-2] |
| `headless.py` job | `param_min` | | {parameter: lower bound}, e.g. -Infinity for a plume part over a fixed ambient column |

The minimum full-resolution transmittance of the plume and ambient layers
over the fit window is kept after every forward-model evaluation
(`Analyser.last_plume_transmittance_min`, `last_atm_transmittance_min`).

`headless.py` fits one window from a JSON job (`python headless.py
job.json`); its docstring lists the job keys and the output columns.

## Tests

    python -m pytest -q tests

The cross-sections are synthetic, so neither RFM nor a line list is
needed.

## Known limitations (not changed)

- In the emission and solar geometries a gas is in one layer only (commit 4
  covers the layer geometry).
- A continuum gas in a layer whose temperature varies needs 21 x 21 RFM
  runs (amounts at each temperature), against 21 when the temperature is
  held.
- `FitResult.rmse` is the mean squared error (sklearn
  `mean_squared_error`), not its root.
- `FitResult.params` is the Analyser's own `Parameters` object, not a copy.
  Read the fitted values of one spectrum before fitting the next.
- RFM's driver and output files are written in the working directory (this
  folder, and `RFM/work/<n>/` for the runs made at the same time), so two
  plumeIR runs must not use one copy of the code at the same time.

## Environment

Python 3.11. RFM v5.21 (Dudhia 2017, eodg.atm.ox.ac.uk/RFM) compiled with
gfortran, as `RFM/source/rfm` (`RFM/source/rfm.exe` on Windows).

```
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation "spc @ git+https://github.com/rohanisaac/spc@f35b90fedb995778ad11b276582d89e233774e5e"
```
