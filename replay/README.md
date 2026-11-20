# Finite replay

This directory holds the deterministic finite replay accompanying the preprint. With Python 3.12, install the pinned dependencies and run from the repository root:

```sh
python -m pip install numpy==2.5.5 scipy==1.18.1 mpmath==1.4.1 python-flint==0.9.0
(cd replay && python -B replay.py)
(cd replay && python -B replay.py --all-mutants)
```

The first command checks the finite exact-rational and outward-rounded interval statements, the polymer budget constants, the boundary oscillator bracket, and a finite numerical sanity scan. It prints an all-checks-passed summary and exits 0. The second command verifies that all 16 guarded mutants are rejected with exit status 1.

`replay.py` is distributed byte-identical to the reviewed replay script, and `SOURCE-PINS.json` records its SHA-256 digest. These finite checks do not formalize the analytic functional analysis, the oscillator trace identities, the infinite cluster expansion, or the canonical thermodynamic transfer. The mathematical claims and their hypotheses are stated in `paper/main.tex`.
