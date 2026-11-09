"""Tests for the fixes on branch v2.0-fixes. They need plumeIR's Python
environment but not RFM: the RFM cross-sections are replaced by synthetic
Gaussian lines with a known column and temperature dependence, so the fits
can be checked against exact answers.

    cd <plumeIR folder> && python -m pytest -q tests
"""
import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)                      # plumeIR resolves ./standards.txt, ./xsec, ./RFM relative to cwd
sys.path.insert(0, ROOT)

from initialise import readConfig            # noqa: E402
import analysis                              # noqa: E402
import customXSC                             # noqa: E402
from analysis import makeILS, Analyser       # noqa: E402
from parameters import Parameters, Geometry, Retrieval  # noqa: E402
from spectra import Spectrum                 # noqa: E402
from unitConversion import ppm2scd           # noqa: E402

N_PER_WN = 25.0


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic cross-sections in place of RFM
# ----------------------------------------------------------------------------------------------------------------------
LINES = {   # species -> (centres [cm-1], strengths [cm2 at 296 K], lower-state energy [cm-1])
    'SO2': ([2492.0, 2500.0, 2508.0, 2515.0, 2522.0], [2e-19, 4e-19, 3e-19, 2e-19, 1e-19], [50, 150, 400, 900, 1500]),
    'N2O': ([2480.0, 2530.0], [5e-19, 3e-19], [100, 300]),
    'CH4': ([2470.0, 2545.0], [1e-19, 1e-19], [200, 200]),
}


def synthetic_xsc(wn, species, temp):
    """Absorption cross-section [cm2 molec-1] of Gaussian lines (HWHM 0.1 cm-1)
    with a Boltzmann temperature dependence of the line strengths."""
    c2 = 1.4388
    sig = np.zeros_like(wn)
    for nu0, s0, elow in zip(*LINES[species]):
        s = s0 * (296.0 / temp) * np.exp(-c2 * elow * (1 / temp - 1 / 296.0))
        sig += s * np.exp(-0.5 * ((wn - nu0) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))
    return sig


def fake_makeGasXSC(wn_start, wn_stop, nper_wn, conc=1.0, temp=298, pres=1013, species='SO2'):
    """Same interface and units as customXSC.makeGasXSC: extinction [m-1] for a
    gas at conc [ppm], temp [K], pres [hPa]."""
    npts = int((wn_stop - wn_start) * nper_wn) + 1
    wn = np.linspace(wn_start, wn_stop, npts)
    n = 7.2429e18 * pres / temp * conc * 1e-6          # molec cm-3
    return {'wn': wn, 'Bext': n * synthetic_xsc(wn, species, temp) * 100.0, 'conc': conc}


@pytest.fixture
def synthetic_rfm(monkeypatch):
    monkeypatch.setattr(analysis, 'makeGasXSC', fake_makeGasXSC)
    monkeypatch.setattr(customXSC, 'makeGasXSC', fake_makeGasXSC)


def make_analyser(tmpdir, fit_gas_temp=True, force_param=None, df_dir=None, fov=0.06, atm_temp=270.0,
                  atm_pres=1020.0, nu_shift_max=1.0, plume_gases='SO2'):
    geometry = Geometry('layer', pathlength=0.3, atm_temp=atm_temp, atm_pres=atm_pres, plume_temp=450.0,
                        plume_pres=atm_pres, plume_thickness=0.05)
    retrieval = Retrieval(type='layer', update_params=True, use_bounds=True, fit_gas_temp=fit_gas_temp,
                          subtract_self=False, no_gas=False, fit_fov=False, fov=fov, fit_max_opd=False,
                          max_opd=2.0, apod_type='Triangular', fit_shift=True, nu_shift=0.0,
                          nu_shift_max=nu_shift_max, force_param=force_param, df_dir=df_dir,
                          model_gas_emission=False, fit_source_temp=False)
    fit = {'polydeg': 3, 'atm_gases': ['N2O', 'CH4'], 'plume_gases': plume_gases, 'plume_aero': None,
           'targets': ['N2O', 'CH4', 'SO2', 'gas_temp'], 'fix_atm_gas': None,
           'fit_gas_temp': fit_gas_temp, 'fit_atm_temp': False, 'model_gas_emission': False}
    params = Parameters()
    params.extract(fit, retrieval, geometry)
    data_dir = os.path.join(tmpdir, 'data') + '/'
    os.makedirs(data_dir, exist_ok=True)
    an = Analyser(params, geometry, [2450.0, 2550.0], retrieval=retrieval, name='FIT1', data_dir=data_dir,
                  force_ref=True)
    return an


def truth(an, so2_scale=3.0, n2o_scale=1.2, ch4_scale=0.8, gas_temp=520.0, nu_shift=0.3):
    """Parameter vector in the order of the fitted parameters, with a smooth
    background of amplitude ~1 (not 1 for every polynomial coefficient)."""
    p = {}
    for name in an.params.poptNamesList():
        p[name] = {'poly0': 2e-8, 'poly1': -1e-4, 'poly2': 0.9, 'N2O': n2o_scale, 'CH4': ch4_scale,
                   'SO2': so2_scale, 'N2O_plume': 2.0, 'gas_temp': gas_temp, 'nu_shift': nu_shift}[name]
    return p


def synth_spectrum(an, p, noise=0.0, seed=0, name='syn_2024_01_14_1820_11_128_sbm.spc'):
    wn = np.arange(2440.0, 2560.0, 0.24)
    pv = [p[n] for n in an.params.poptNamesList()]
    y = an.forwardTransmission(wn[(wn >= 2450) & (wn <= 2550)], *pv)
    # the forward model only returns the fit window; embed it in a wider spectrum
    full = np.full_like(wn, np.nan)
    full[(wn >= 2450) & (wn <= 2550)] = y
    full = np.where(np.isnan(full), np.nanmean(y), full)
    if noise:
        full = full + np.random.default_rng(seed).normal(0, noise * np.nanmean(y), len(full))
    return Spectrum(wn, full, spec_type='sbm', dtime=pd.Timestamp('2024-01-14 18:20:11'), fname=name)


# ----------------------------------------------------------------------------------------------------------------------
# Line-shape kernel
# ----------------------------------------------------------------------------------------------------------------------
def fwhm(k, g):
    h = k.max() / 2
    i = int(np.argmax(k))
    return np.interp(h, k[i:][::-1], g[i:][::-1]) - np.interp(h, k[:i + 1], g[:i + 1])


@pytest.mark.parametrize('L', [0.517, 0.531, 0.637, 0.877, 1.0, 1.8, 2.0])
def test_ils_width_matches_opd(L):
    """Triangular apodisation: FWHM of sinc^2 is 0.8859/L. The published code was
    3-10 % too wide for L not a multiple of 0.08 cm."""
    o = makeILS(fov=0.01, nper_wn=N_PER_WN, wn=2500.0, max_opd=L, apod_type='Triangular')
    assert abs(fwhm(o['kernel'], o['wn_grid']) / (0.8859 / L) - 1) < 0.01


@pytest.mark.parametrize('fov,wn', [(0.025, 2500.0), (0.06, 2500.0), (0.08, 2760.0), (0.08, 4040.0), (0.1, 4040.0)])
def test_ils_fov_shift_sum_and_symmetry(fov, wn):
    """Kernel sums to 1 after truncation, has an odd length with wn = 0 in the
    middle, and its centroid is the self-apodisation shift -wn fov^2 / 16."""
    o = makeILS(fov=fov, nper_wn=N_PER_WN, wn=wn, max_opd=2.0, apod_type='Triangular')
    k, g = o['kernel'], o['wn_grid']
    assert abs(k.sum() - 1) < 1e-9
    assert len(k) % 2 == 1 and abs(g[len(g) // 2]) < 1e-9
    assert abs((k * g).sum() - (-wn * fov ** 2 / 16)) < 0.012
    assert abs(np.diff(g).mean() - 1 / N_PER_WN) < 1e-9


@pytest.mark.parametrize('fov,wn', [(0.15, 2500.0), (0.2, 4100.0), (0.3, 2085.0), (0.5, 4900.0)])
def test_ils_wide_fov(fov, wn):
    """A FOV box wider than the default kernel grid (+-12.5 cm-1) is neither cut
    nor replaced: the kernel still sums to 1, is centred on -wn fov^2 / 16 and
    has the spread of the box (width wn fov^2 / 8) and the ILS together."""
    o = makeILS(fov=fov, nper_wn=N_PER_WN, wn=wn, max_opd=2.0, apod_type='Triangular')
    k, g = o['kernel'], o['wn_grid']
    c = (k * g).sum()
    box = wn * fov ** 2 / 8
    ils_var = (k0 := makeILS(fov=0.0, nper_wn=N_PER_WN, wn=wn, max_opd=2.0, apod_type='Triangular'))['kernel'] \
        @ k0['wn_grid'] ** 2
    assert abs(k.sum() - 1) < 1e-9
    assert len(k) % 2 == 1 and abs(g[len(g) // 2]) < 1e-9
    assert abs(c - (-box / 2)) < 0.012
    # 3 %: the second moment of the sinc^2 tails depends on where the kernel is cut
    assert abs(np.sqrt(k @ (g - c) ** 2) / np.sqrt(box ** 2 / 12 + ils_var) - 1) < 0.03


def test_ils_continuous_in_fov_and_opd():
    """fit_fov and fit_max_opd need a kernel that changes smoothly with both."""
    w = [fwhm(*[makeILS(fov=f, nper_wn=N_PER_WN, wn=2500.0, max_opd=2.0, apod_type='Triangular')[k]
                for k in ('kernel', 'wn_grid')]) for f in np.linspace(0.06, 0.07, 11)]
    assert np.all(np.diff(w) > 0) and max(np.diff(w)) < 3 * min(np.diff(w))
    w = [fwhm(*[makeILS(fov=f, nper_wn=N_PER_WN, wn=4100.0, max_opd=2.0, apod_type='Triangular')[k]
                for k in ('kernel', 'wn_grid')]) for f in np.linspace(0.12, 0.16, 21)]    # across the old grid edge
    assert np.all(np.diff(w) > 0) and max(np.diff(w)) < 3 * min(np.diff(w))
    w = [fwhm(*[makeILS(fov=0.01, nper_wn=N_PER_WN, wn=2500.0, max_opd=L, apod_type='Triangular')[k]
                for k in ('kernel', 'wn_grid')]) for L in np.linspace(1.9, 2.1, 11)]
    assert np.all(np.diff(w) < 0) and max(-np.diff(w)) < 3 * min(-np.diff(w))


def test_fov_term_uses_window_centre(synthetic_rfm, tmp_path):
    """With the FOV evaluated at the window centre, a 0.08 rad FOV broadens the
    model lines by about 2 cm-1 at 2500 cm-1; at half the window width (the
    published bug) it would do nothing."""
    an_wide = make_analyser(str(tmp_path), fov=0.08)
    an_narrow = make_analyser(str(tmp_path), fov=0.01)
    p = truth(an_wide)
    y_w = an_wide.forwardTransmission(np.arange(2450.0, 2550.0, 0.24), *[p[n] for n in an_wide.params.poptNamesList()])
    y_n = an_narrow.forwardTransmission(np.arange(2450.0, 2550.0, 0.24), *[p[n] for n in an_narrow.params.poptNamesList()])
    depth_w, depth_n = 1 - y_w.min() / np.median(y_w), 1 - y_n.min() / np.median(y_n)
    assert depth_w < 0.6 * depth_n


# ----------------------------------------------------------------------------------------------------------------------
# Forward model and fit: closure with synthetic cross-sections
# ----------------------------------------------------------------------------------------------------------------------
def test_closure_columns_and_temperature(synthetic_rfm, tmp_path):
    """A spectrum made by plumeIR's own forward model is fitted back to the
    columns and plume temperature it was made with."""
    an = make_analyser(str(tmp_path))
    p = truth(an)
    sp = synth_spectrum(an, p, noise=2e-4)
    r = an.fitSpectrum(sp, update_params=False, use_bounds=True)
    assert r.nerr == 1
    for gas, scale in (('SO2', p['SO2']), ('N2O', p['N2O']), ('CH4', p['CH4'])):
        assert abs(r.params[gas].fit_val / scale - 1) < 2e-3, gas
    assert abs(r.params['gas_temp'].fit_val - p['gas_temp']) < 3.0
    assert abs(r.params['nu_shift'].fit_val - p['nu_shift']) < 0.01


def test_ambient_columns_at_site_conditions(synthetic_rfm, tmp_path):
    """The reported ambient-layer column is fit_val * ref_column with ref_column
    at the configured T and P; the cross-sections have to be made at the same
    T and P, or the column is off by n(T,P)/n(298 K, 1013 hPa) (13 % for
    270 K and 1020 hPa). Checked against a column computed independently."""
    an = make_analyser(str(tmp_path), atm_temp=270.0, atm_pres=1020.0)
    p = truth(an)
    sp = synth_spectrum(an, p)
    r = an.fitSpectrum(sp, update_params=False, use_bounds=True)
    # independent reconstruction of the N2O column that produced the spectrum
    std_conc = an.reference['atm']['ref_column']['N2O'] / ppm2scd(1.0, 0.3, temp=270.0, pres=1020.0)
    col_true = ppm2scd(std_conc * p['N2O'], 0.3, temp=270.0, pres=1020.0)
    col_fit = r.params['N2O'].fit_val * r.params['N2O'].ref_column
    assert abs(col_fit / col_true - 1) < 2e-3
    # and the optical depth of the reference really is column x sigma(270 K)
    wn = an.model_grid
    od = an.reference['atm']['N2O']
    expect = an.reference['atm']['ref_column']['N2O'] * synthetic_xsc(wn, 'N2O', 270.0)
    assert np.allclose(od, expect, rtol=1e-4)   # 7.2429e18 vs plumeIR's R and N_A


def test_gas_in_both_layers(synthetic_rfm, tmp_path):
    """A gas listed as ambient and plume gets a second parameter '<gas>_plume'
    at the plume temperature; both columns are recovered (v2.0 silently kept
    only the plume entry)."""
    an = make_analyser(str(tmp_path), plume_gases=['SO2', 'N2O'])
    assert 'N2O' in an.params and 'N2O_plume' in an.params
    assert an.params['N2O'].atm_gas and an.params['N2O_plume'].plume_gas
    assert 'N2O_plume' in an.reference['plume']['OD_calib']       # temperature-calibrated
    p = truth(an)
    sp = synth_spectrum(an, p, noise=1e-4)
    r = an.fitSpectrum(sp, update_params=False, use_bounds=True)
    assert r.nerr == 1
    assert abs(r.params['N2O'].fit_val / p['N2O'] - 1) < 0.02
    assert abs(r.params['N2O_plume'].fit_val / p['N2O_plume'] - 1) < 0.05
    assert abs(r.params['SO2'].fit_val / p['SO2'] - 1) < 5e-3


def test_plume_temperature_bounds(synthetic_rfm, tmp_path):
    """The plume temperature bounds are the calibration range of the
    cross-sections (constants.gas_temps, 200-800 K), below any air temperature."""
    an = make_analyser(str(tmp_path), atm_temp=268.0)
    assert (an.params['gas_temp'].min, an.params['gas_temp'].max) == (200, 800)


def test_force_param_survives_a_failed_fit(synthetic_rfm, tmp_path):
    """gas_temp imposed from a previous fit: p0 and bounds have the same length,
    the imposed value is used, and a failed spectrum does not break the fits
    that follow (it did in v2.0)."""
    an0 = make_analyser(str(tmp_path))
    p = truth(an0)
    names = ['a_2024_01_14_1820_11_128_sbm.spc', 'b_2024_01_14_1820_15_060_sbm.spc', 'c_2024_01_14_1820_19_038_sbm.spc']
    temps = [480.0, 500.0, 530.0]
    force = os.path.join(str(tmp_path), 'force')
    os.makedirs(force)
    pd.DataFrame({'File': names, 'gas_temp': temps}).to_csv(os.path.join(force, 'fitResults_SO2_ALL.csv'), index=False)
    an = make_analyser(str(tmp_path), force_param='gas_temp', df_dir=force)
    assert len(an.p0) == len(an.bounds[0]) == len(an.bounds[1])
    assert 'gas_temp' not in an.params.poptNamesList()
    results = []
    for i, (name, T) in enumerate(zip(names, temps)):
        q = dict(p, gas_temp=T)
        sp = synth_spectrum(an0, q, name=name)
        if i == 1:                                   # corrupt samples in the window: the fit must fail cleanly
            x, y = sp.spectrum
            y = y.copy()
            y[(x > 2495) & (x < 2505)] = np.nan
            sp.update(spectrum=np.vstack([x, y]))
        r = an.fitSpectrum(sp, update_params=True, use_bounds=True)
        # FitResult.params is the Analyser's own Parameters object (not a copy),
        # so values have to be read before the next fit
        results.append((r.nerr, r.params['gas_temp'].fit_val, r.params['SO2'].fit_val))
    assert [r[0] for r in results] == [1, 0, 1]
    assert results[0][1] == pytest.approx(480.0) and results[2][1] == pytest.approx(530.0)
    assert abs(results[2][2] / p['SO2'] - 1) < 2e-3


# ----------------------------------------------------------------------------------------------------------------------
# RFM call guarded
# ----------------------------------------------------------------------------------------------------------------------
def test_rfm_failure_raises(monkeypatch, tmp_path):
    """A failed RFM run must raise instead of returning the previous output file."""
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        os.makedirs('RFM/output')
        with open('RFM/output/opt00100.out', 'w') as f:        # stale file from an earlier call
            f.write('!\n!\n!\n 10 2400.0 0.04 2600.0 \'Optical Depth\'\n' + ' '.join(['1.0'] * 5001) + '\n')

        class Proc:
            returncode = 0

        def no_rfm(*a, **k):                                   # RFM that produces nothing
            with open('rfm.log', 'w') as f:
                f.write('F-RFM: some fatal error\n')
            return Proc()
        monkeypatch.setattr(customXSC, 'run', no_rfm)
        with pytest.raises(RuntimeError):
            customXSC.makeGasXSC(2400.0, 2600.0, 25.0, conc=1.0, temp=300.0, pres=1000.0, species='SO2')
        assert not os.path.exists('RFM/output/opt00100.out')   # stale file removed before the run
    finally:
        os.chdir(cwd)


# ----------------------------------------------------------------------------------------------------------------------
# Config parsing
# ----------------------------------------------------------------------------------------------------------------------
def test_readconfig_windows_path_and_lists(tmp_path):
    cfg = tmp_path / 'c.txt'
    cfg.write_text("*DATA\ndata_dir : C:/Users/x/data/\n*FIT1\nfit : True\nfit_window : 2450, 2550\n"
                   "atm_gases : N2O, H2O, CH4\nplume_gases : SO2\nplume_aero : None\npolydeg : 3\n")
    c = readConfig(str(cfg))
    assert c['DATA']['data_dir'] == 'C:/Users/x/data/'
    assert c['FIT1']['fit_window'] == [2450.0, 2550.0]
    assert c['FIT1']['atm_gases'] == ['N2O', 'H2O', 'CH4'] and c['FIT1']['plume_gases'] == 'SO2'
    assert c['FIT1']['plume_aero'] is None and c['FIT1']['fit'] is True


def test_read_midac_unknown_ylabel_uses_filename(tmp_path, monkeypatch):
    """An SPC y-label plumeIR does not know falls back to the file name."""
    import types
    import spectra as sp

    class FakeSub:
        y = np.linspace(1, 2, 10)

    class FakeSpc:
        ylabel = 'Arbitrary or Single Beam with Valley Peaks'
        x = np.linspace(2000, 2100, 10)
        sub = [FakeSub()]
        log_dict = {}

    monkeypatch.setattr(sp.spc, 'File', lambda fname: FakeSpc())
    s = sp.read_midac(str(tmp_path / '2025_07_31_1629_34_714_sbm.spc'))
    assert s.spec_type == 'sbm'
    with pytest.raises(ValueError):
        sp.read_midac(str(tmp_path / 'mystery.spc'))


def test_rfm_executable(monkeypatch, tmp_path):
    """rfm.exe where it is the build present or on Windows, else rfm."""
    import rfmexe
    base = tmp_path / "RFM" / "source" / "rfm"
    base.parent.mkdir(parents=True)
    monkeypatch.setattr(rfmexe, "_BASE", str(base))
    assert rfmexe.rfm_executable() == str(base)          # nothing built: the plain name (RFM reports the error)
    base.with_suffix(".exe").write_text("")
    assert rfmexe.rfm_executable() == str(base) + ".exe"  # only the Windows build present
    base.write_text("")
    monkeypatch.setattr(rfmexe.os, "name", "posix")
    assert rfmexe.rfm_executable() == str(base)
    monkeypatch.setattr(rfmexe.os, "name", "nt")
    assert rfmexe.rfm_executable() == str(base) + ".exe"


def test_apriori_background_from_spectrum():
    """np.polyfit coefficients (highest degree first) give back the
    background np.polyval evaluates in the forward model."""
    x = np.linspace(2450, 2550, 501)
    bg = 3.0 * x - 2000.0
    y = bg * (1 - 0.3 * np.exp(-((x - 2500) / 5) ** 2))
    c = np.polyfit(x, y, 1)
    assert np.allclose(np.polyval(c, x) / bg, 1, atol=0.08)


def test_background_on_measurement_grid():
    """The saved background is the polynomial convolved with the ILS as
    in the forward model; for a straight line that differs from the bare
    polynomial by < 0.1 % (the FOV makes the ILS slightly asymmetric)."""
    import initialise  # noqa: F401  (analysis and initialise import each other)
    import analysis
    x = np.arange(2440.0, 2560.0, 0.04)
    line = 3.0 * x - 2000.0
    k = analysis.makeILS(fov=0.05, max_opd=1.8, nper_wn=25, wn=2500.0)["kernel"]
    conv = np.convolve(line, k, "same")
    inner = slice(len(k), -len(k))                       # away from the edges of the 'same' convolution
    assert np.allclose(conv[inner], line[inner], rtol=1e-3)


def test_fov_bounds_from_retrieval_settings():
    """fov_min and fov_max (RETRIEVAL keys) bound a fitted FOV; the defaults
    are plumeIR's 0.01 and 0.15, 'inf' removes the upper bound."""
    assert (Retrieval().fov_min, Retrieval().fov_max) == (0.01, 0.15)
    r = Retrieval(fov=0.06, fov_min=0.0, fov_max=float('inf'))
    assert (r.fov_min, r.fov_max) == (0.0, float('inf'))


def test_parallel_processes_and_rfm_guard():
    """At most one process per MIN_CHUNK spectra; a spawned worker
    process cannot run RFM (it shares the folder RFM writes in)."""
    import headless
    assert headless.number_of_processes(7, 85) == 7
    assert headless.number_of_processes(7, 25) == 2
    assert headless.number_of_processes(7, 5) == 1
    assert headless.number_of_processes(1, 400) == 1
    with pytest.raises(RuntimeError, match="would run RFM"):
        headless._no_rfm(["rfm"])


def test_worker_that_cannot_start_ends_the_run(monkeypatch):
    """_init_worker installs the RFM guard and keeps an initialisation
    error, which the first chunk raises (an initializer that raises makes the
    pool restart the process without end)."""
    import headless
    import customXSC
    real = customXSC.run

    def failing(job, files):
        assert analysis.run is headless._no_rfm and customXSC.run is headless._no_rfm   # guard installed first
        raise FileNotFoundError("layerReference.pkl")
    monkeypatch.setattr(headless, "WindowFit", failing)
    monkeypatch.setattr(headless, "_INIT_ERROR", None)
    try:
        headless._init_worker({}, [])
        with pytest.raises(RuntimeError, match="could not start: FileNotFoundError"):
            headless._run_chunk([0])
    finally:
        analysis.run = customXSC.run = real


def test_reference_identity_includes_rfm(monkeypatch, tmp_path):
    """The reference header's code identity changes with the RFM executable,
    so references computed by another RFM build are not reused."""
    a = tmp_path / "rfm_a"
    a.write_bytes(b"one build")
    b = tmp_path / "rfm_b"
    b.write_bytes(b"another build")
    monkeypatch.setattr(analysis, "rfm_executable", lambda: str(a))
    ida = analysis.codeId()
    monkeypatch.setattr(analysis, "rfm_executable", lambda: str(b))
    assert analysis.codeId() != ida


def test_saturation_diagnostic_inside_the_fit_window(synthetic_rfm, tmp_path, monkeypatch):
    """The minimum plume transmittance is taken over the fit window, not over
    the model padding around it: a saturated line at 2420 cm-1, outside the
    2450-2550 window, does not change it."""
    def minimum(folder):
        an = make_analyser(str(folder))
        p = truth(an)
        an.forwardTransmission(np.arange(2450.0, 2550.0, 0.24), *[p[n] for n in an.params.poptNamesList()])
        return an.last_plume_transmittance_min
    plain = minimum(tmp_path / "a")
    so2 = LINES['SO2']
    monkeypatch.setitem(LINES, 'SO2', ([2420.0] + so2[0], [5e-16] + so2[1], [50] + so2[2]))
    assert minimum(tmp_path / "b") == pytest.approx(plain, rel=1e-9)


def test_progress_file(tmp_path):
    """The progress file: the number of spectra, then one line per fitted
    spectrum; without the job key nothing is written."""
    with open(os.path.join(ROOT, "headless.py"), encoding="utf-8") as fh:
        src = fh.read()
    ns = {}
    start = src.index("def progress_start")
    end = src.index("class WindowFit")
    exec(src[start:end], ns)
    f = tmp_path / "progress.txt"
    ns["progress_start"](str(f), 3)
    for name in ("a.spc", "b.spc"):
        ns["progress_add"](str(f), name)
    assert f.read_text(encoding="utf-8").splitlines() == ["3", "a.spc", "b.spc"]
    ns["progress_start"](None, 3)
    ns["progress_add"](None, "c.spc")


def test_held_temperature_needs_one_cross_section(synthetic_rfm, tmp_path, monkeypatch):
    """A layer temperature that is neither fitted nor imposed: one cross-section
    per gas at that temperature (degree-0 calibration, exact); a fitted or
    imposed one: the calibration over the range."""
    calls = []
    def counting(*a, **k):
        calls.append(k.get('temp'))
        return fake_makeGasXSC(*a, **k)
    monkeypatch.setattr(analysis, 'makeGasXSC', counting)
    an = make_analyser(str(tmp_path), fit_gas_temp=False)
    ref = an.reference
    assert ref['atm']['OD_calib']['N2O'].shape[0] == 1 and ref['plume']['OD_calib']['SO2'].shape[0] == 1
    assert np.allclose(np.polyval(ref['plume']['OD_calib']['SO2'], 612.0), ref['plume']['SO2'])
    assert len(calls) == 3                                   # N2O, CH4, SO2: one each
    calls.clear()
    an = make_analyser(str(tmp_path / 'fitted'), fit_gas_temp=True)
    assert an.reference['plume']['OD_calib']['SO2'].shape[0] == 11      # degree 10 over gas_temps
    assert an.reference['atm']['OD_calib']['N2O'].shape[0] == 1         # ambient temperature held
    assert len(calls) == 2 + 21 + 1
