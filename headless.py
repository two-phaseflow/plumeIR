"""Batch fit of one plumeIR fit window without the plotting and file-writing
parts of plumeIR.py.

    python headless.py job.json

The run takes place in this folder (plumeIR resolves ./RFM, ./atm, ./xsec and
writes rfm.drv here), so two runs must not share a folder at the same time.

job.json keys
  config            plumeIR config file (one *FIT section)
  files             spectrum files in time order (POSIX paths, absolute or
                    relative to this folder)
  out_csv           results, one row per spectrum (see below)
  all_csv           plumeIR-style summary ('*fitResults*_ALL.csv') that a later
                    run can impose parameters from with force_param
  select_brightest  optional: fit only the N spectra with the highest mean
                    signal in signal_window [lo, hi] cm-1 (line-shape
                    calibration)
  wn_scale          optional: multiply each wavenumber axis by this before the
                    fit (default 1.0)
  save_fits         optional: indices (into files) whose measured and modelled
                    spectra are written to fits_npz
  workers           optional: number of processes (default 1). The time-ordered
                    files are split into that many consecutive chunks of at
                    least MIN_CHUNK spectra, fitted in parallel. This process
                    initialises first (builds the reference cross-sections or
                    loads them from the data folder); the others then inherit it
                    (fork, Linux and macOS) or initialise again from the saved
                    references (spawn, Windows), and must not run RFM: a worker
                    that would, or that fails to start, ends the run with an error. Rows come back in input
                    order. With update_params False every spectrum starts from
                    the a priori values, so the results do not depend on the
                    number of processes. With update_params each spectrum starts
                    from the previous fit, and the first spectrum of every chunk
                    from the a priori values, as the first of the series does.
  progress          optional: a file this run writes its progress to: first
                    the number of spectra to fit, then one line per fitted
                    spectrum, from every process (progress_start, progress_add)
The a priori background polynomial of each fit that does not start from the
previous one is a least-squares fit of that spectrum (plumeIR sets every
coefficient to 1); a fit from the previous values that fails is repeated once
from the a priori values. out_csv column "start" records which.

out_csv columns: File, Time, nerr (1 = fit converged), MSE (plumeIR's
FitResult.rmse is sklearn's mean_squared_error, i.e. the mean squared
error), R2, rmse_rel (sqrt(MSE) / mean measured signal), mean_signal, plume_T_min and
atm_T_min (minimum transmittance of the fitted plume and ambient layers at
full resolution, the saturation diagnostic); for
each gas atm:GAS or plume:GAS and its _err, as slant column density
[molec cm-2] (fitted scaling times plumeIR's ref_column); every other fitted
or fixed parameter (gas_temp, fov, nu_shift, max_opd, poly0, poly1, ...) and
its _err. fits_npz holds, per saved spectrum i, i_wn, i_meas, i_model and
i_bg (the fitted background on the same grid).

Initialisation, reference cross-sections, forward model and fit are plumeIR's
own (initialiseFit, Analyser.fitSpectrum).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
if os.path.isdir(os.path.join(HERE, "shim")):            # pyspectra stand-in for older code
    sys.path.insert(0, os.path.join(HERE, "shim"))

import matplotlib                                         # noqa: E402

matplotlib.use("Agg")
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from initialise import initialiseFit                      # noqa: E402
from spectra import read_spectrum                         # noqa: E402


MIN_CHUNK = 10          # spectra per process at least: a process costs one initialisation


def progress_start(path, n):
    """Progress file of a run (job key "progress"): first line the number of
    spectra to fit; every fitted spectrum then adds a line (progress_add).
    A file, not the console, because spawned processes (Windows) do not
    write to the console of the run that started them."""
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{n}\n")


def progress_add(path, name):
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(name + "\n")


class WindowFit:
    """One fit window, initialised once per process: plumeIR's analyser, the
    spectra to fit, and the fit of one spectrum or a chunk of spectra."""

    def __init__(self, job, files=None):
        t0 = time.time()
        config, geometry, params, retrieval, analysers = initialiseFit(job["config"], force_ref=False)
        if len(analysers) != 1:
            raise SystemExit("headless.py fits one window per run (one *FIT section)")
        an = analysers[0]
        # p0 and bounds of the parameters that vary (as Analyser.__init__ builds
        # them in this fork; repeated here for code without that fix)
        an.p0 = an.params.fittedValuesList()
        an.bounds = an.params.bounds()
        self.an, self.retrieval = an, retrieval
        self.scale = float(job.get("wn_scale", 1.0))
        self.save_idx = set(job.get("save_fits", []))
        self.progress = job.get("progress")
        self.p0_apriori = list(an.p0)
        varying = [n for n, p in an.params.items() if p.vary]
        self.poly_idx = [k for k, n in enumerate(varying) if n.startswith("poly")]
        self.files = list(files) if files is not None else self.select(job)
        print(f"init {time.time() - t0:.0f} s", flush=True)

    def load(self, fname):
        sp = read_spectrum(fname)
        if self.scale != 1.0:
            x, y = sp.spectrum
            sp.update(spectrum=np.vstack([x * self.scale, y]))
        return sp

    def select(self, job):
        """The job's files, or the N brightest of them (select_brightest)."""
        files = list(job["files"])
        if job.get("select_brightest"):
            lo, hi = job["signal_window"]
            sig = []
            for f in files:
                x, y = self.load(f).spectrum
                sig.append(float(np.mean(y[(x >= lo) & (x <= hi)])))
            keep = set(np.argsort(sig)[::-1][: int(job["select_brightest"])].tolist())
            files = [f for k, f in enumerate(files) if k in keep]
            print(f"brightest {len(files)} of {len(sig)} spectra selected", flush=True)
        return files

    def apriori(self, sp):
        """plumeIR's a priori values, with the background polynomial from a
        least-squares fit of the measured spectrum in the window instead of
        every coefficient at 1 (a background of about the wavenumber, often far
        from the signal, from which strongly absorbing spectra did not converge)."""
        p0 = list(self.p0_apriori)
        if self.poly_idx:
            x, y = self.an.preProcess(sp).spectrum
            for k, c in zip(self.poly_idx, np.polyfit(x, y, len(self.poly_idx) - 1)):   # highest degree first
                p0[k] = float(c)
        return p0

    def background(self, r):
        """The fitted background on the measurement grid, computed as the
        forward model computes the spectrum (Analyser.forwardTransmission) with
        every transmittance set to 1: polynomial on the model grid, convolved
        with the fitted ILS, plus offset, shifted by nu_shift and interpolated."""
        from analysis import makeILS
        from scipy.interpolate import griddata
        an = self.an
        v = {n: q.fit_val for n, q in r.params.items()}
        coef = [v[f"poly{k}"] for k in range(sum(1 for n in v if n.startswith("poly")))]
        mg = np.asarray(an.model_grid, float)
        wn = np.asarray(r.grid, float)
        bg = np.polyval(coef, mg) if coef else np.ones_like(mg)
        kernel = makeILS(fov=v["fov"], nper_wn=an.nper_wn, wn=(wn.max() + wn.min()) / 2, max_opd=v["max_opd"],
                         apod_type=an.apod_type)["kernel"]
        f = np.convolve(bg, kernel, "same") + v.get("offset", 0.0)
        return griddata(mg + v.get("nu_shift", 0.0), f, wn, method="cubic")

    def fit_chunk(self, idx):
        """Fit files[idx] in order. A spectrum starts from the previous fit when
        update_params is set and that fit converged, else from the a priori
        values (apriori); a fit from the previous values that fails is repeated
        once from the a priori values."""
        carried = False
        out = []
        for i in idx:
            res, ok = self.fit_one(i, carried)
            carried = bool(self.retrieval.update_params) and ok
            out.append(res)
        return out

    def fit_one(self, i, carried):
        an, retrieval, fname = self.an, self.retrieval, self.files[i]
        sp = self.load(fname)
        if not carried:
            an.p0 = self.apriori(sp)
        start = "previous" if carried else "a priori"
        r = an.fitSpectrum(sp, update_params=retrieval.update_params, use_bounds=retrieval.use_bounds)
        if r.nerr != 1 and carried:
            an.p0 = self.apriori(sp)
            start = "a priori (retry)"
            r = an.fitSpectrum(sp, update_params=retrieval.update_params, use_bounds=retrieval.use_bounds)
        base = os.path.basename(fname)
        row = {"File": base, "Time": sp.dtime, "nerr": r.nerr, "start": start, "MSE": r.rmse, "R2": r.r2,
               "rmse_rel": float(np.sqrt(r.rmse) / np.mean(r.spec)) if r.nerr else np.nan,
               "mean_signal": float(np.mean(r.spec)),
               # minimum full-resolution transmittance of the fitted plume and ambient
               # layers in the window (FitResult evaluates the model at popt last)
               "plume_T_min": getattr(an, "last_plume_transmittance_min", np.nan) if r.nerr else np.nan,
               "atm_T_min": getattr(an, "last_atm_transmittance_min", np.nan) if r.nerr else np.nan}
        row_all = {"File": base, "Time": sp.dtime}
        for name, p in r.params.items():
            if p.atm_gas or p.plume_gas:
                layer = "atm" if p.atm_gas else "plume"
                row[f"{layer}:{name}"] = p.fit_val * p.ref_column
                row[f"{layer}:{name}_err"] = p.fit_err * p.ref_column
                row_all[f"{name}_val"] = p.fit_val
                row_all[f"{name}_scd [molec.cm^-2]"] = p.fit_val * p.ref_column
            else:
                row[name] = p.fit_val
                row[f"{name}_err"] = p.fit_err
                if not name.startswith("poly"):              # the force table keeps plumeIR's columns
                    row_all[name] = p.fit_val
        fit = {}
        if i in self.save_idx and r.nerr:
            fit = {f"{i}_wn": r.grid, f"{i}_meas": r.spec, f"{i}_model": r.model,
                   f"{i}_bg": self.background(r)}
        if i % 10 == 0:
            print(i, base, f"R2 {r.r2:.4f}", flush=True)
        progress_add(self.progress, base)
        return (row, row_all, fit), r.nerr == 1


def number_of_processes(requested, n_files):
    """Processes for n_files spectra: the number asked for, at most one per
    MIN_CHUNK spectra, at least 1."""
    return max(1, min(int(requested), n_files // MIN_CHUNK))


def main(job):
    t0 = time.time()
    window = WindowFit(job)                    # builds the reference cross-sections or loads them
    files = window.files
    progress_start(job.get("progress"), len(files))
    workers = number_of_processes(job.get("workers", 1), len(files))
    chunks = [c.tolist() for c in np.array_split(np.arange(len(files)), workers)]
    if workers == 1:
        results = window.fit_chunk(chunks[0])
    else:
        import multiprocessing as mp
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        method = os.environ.get("PLUMEIR_START_METHOD", method)     # tests: spawn on Linux as on Windows
        global _WINDOW
        _WINDOW = window                       # what forked processes inherit
        ctx = mp.get_context(method)
        init = {} if method == "fork" else {"initializer": _init_worker, "initargs": (job, files)}
        with ctx.Pool(workers, **init) as pool:
            results = [x for part in pool.map(_run_chunk, chunks) for x in part]
        print(f"{workers} processes ({method}), chunks of {[len(c) for c in chunks]} spectra", flush=True)
    rows = [x[0] for x in results]
    rows_all = [x[1] for x in results]
    saved = {k: v for x in results for k, v in x[2].items()}
    pd.DataFrame(rows).to_csv(job["out_csv"], index=False)
    pd.DataFrame(rows_all).to_csv(job["all_csv"], index=False)
    if saved:
        np.savez(job["fits_npz"], **saved)
    print(f"done {len(rows)} spectra in {time.time() - t0:.0f} s", flush=True)


_WINDOW = None


def _no_rfm(*args, **kwargs):
    raise RuntimeError("a worker process would run RFM: the reference cross-sections it needs were not "
                       "saved by the first process")


_INIT_ERROR = None


def _init_worker(job, files):
    """Start of a spawned process: plumeIR's initialisation from the reference
    cross-sections the first process saved. RFM is disabled here, because RFM
    writes its files in this folder, which every process shares. An error is
    kept and raised by _run_chunk: raised here, it would make the pool start
    the process again without end."""
    global _WINDOW, _INIT_ERROR
    try:
        import analysis
        import customXSC
        analysis.run = customXSC.run = _no_rfm
        _WINDOW = WindowFit(job, files)
    except BaseException as e:                     # noqa: BLE001  (reported by _run_chunk)
        _INIT_ERROR = f"{type(e).__name__}: {e}"


def _run_chunk(idx):
    if _INIT_ERROR is not None:
        raise RuntimeError(f"worker process could not start: {_INIT_ERROR}")
    return _WINDOW.fit_chunk(idx)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        main(json.load(f))
