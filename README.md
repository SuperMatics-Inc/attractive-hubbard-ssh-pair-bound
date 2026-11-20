# An explicit pair bound for the attractive Hubbard model with bond SSH coupling

Davis Rens  
SuperMatics Inc.  
[ORCID 0000-0001-5839-5197](https://orcid.org/0000-0001-5839-5197)  
Version 1.0.0

Paper DOI: [10.5281/zenodo.24406987](https://doi.org/10.5281/zenodo.24406987). Supplement DOI: [10.5281/zenodo.24406989](https://doi.org/10.5281/zenodo.24406989).  
Source repository: https://github.com/SuperMatics-Inc/attractive-hubbard-ssh-pair-bound

## Scope and evidence

The preprint proves a frequency-uniform exponential bound on equal-time on-site pair correlations for the spinful attractive Hubbard model with one harmonic oscillator on each nearest-neighbor bond, on square tori of side at least four. Under the restrictions `beta t^2/U_eff <= 2^-27` and `beta g^2/K <= 2^-13`, the pair correlation decays as `(1965840/475579)(48645/524224)^d` in graph distance and the row sum stays below 3, uniformly in the chemical potential, the torus size, and every positive oscillator frequency. The estimate transfers to thermodynamic canonical sequences at all total fillings, and the pair-order temperature obeys `k_B T_pair <= 2^27 (t^2/U_eff + g^2/K)`.

These are mathematical results for the stated model. They are not evidence for pair order, not a material prediction, and not a measurement. The finite replay checks finite algebra, exact constants, and guarded mutants only; it does not formalize the analytic functional analysis, the oscillator trace identities, the infinite cluster expansion, or the canonical thermodynamic transfer. This preprint has not been peer reviewed.

## Contents and replay

- `paper/` holds the manuscript source, its bibliography, and the built PDF (`main.pdf`).
- `replay/` holds the deterministic finite replay and its source pins.
- `ADMITTED.sha256` records the raw-byte hashes of the admitted source snapshots at the review revision. The released manuscript is one revision later: the review findings and the DOI insertion are documented in the release record, and every change is textual.
- `MANIFEST.sha256` pins each file in this repository except itself.

With Python 3.12+, install the pinned replay dependencies and run both modes:

```sh
python -m pip install numpy==2.5.5 scipy==1.18.1 mpmath==1.4.1 python-flint==0.9.0
(cd replay && python -B replay.py)
(cd replay && python -B replay.py --all-mutants)
sha256sum -c MANIFEST.sha256
```

The first command checks the exact rational and outward-rounded interval statements and the finite numerical sanity scans. The second verifies rejection of all 16 guarded mutants. The manuscript can be rebuilt from `paper/` with Tectonic 0.17.0:

```sh
cd paper && tectonic --keep-logs --outdir ../build main.tex
```

The GitHub Actions workflow checks the source manifest and the replay pins, runs the base and mutant replay modes, verifies the manuscript prose and links, and rebuilds the PDF to compare its text with the committed artifact.

## Licenses and provenance

The manuscript text in `paper/` is distributed under CC-BY-4.0 through the paper record. The replay code and repository documentation use Apache-2.0. The Zenodo supplement is software licensed under Apache-2.0.

AI agents generated the proofs, code, and manuscript drafts under D. Rens' direction. Separate AI agents performed independent reviews.
