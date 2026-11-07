# Release and archival procedure

The public repository is https://github.com/Fanyd271/ST-GRAPE. It is independent
of the private development repository and contains no private Git history.

1. Review the changes, licenses, citation metadata and immutable data manifest.
2. Run the checks in `CONTRIBUTING.md`, then commit and push to this repository.
3. Check the Linux/Windows results at https://github.com/Fanyd271/ST-GRAPE/actions.
4. Build the wheel/source archive with `python -m build`, then run
   `python scripts/build_release.py` to create the complete ZIP and checksums.
5. Tag the reviewed version and attach the ZIP, wheel, source archive and
   `SHA256SUMS` to its GitHub Release. Never replace an existing published tag
   with different numerical results.
6. Archive the exact tagged release in a suitable long-term research repository
   if a persistent software/data DOI is desired. Explain the split MIT/code and
   CC BY 4.0/data licenses in its metadata. Do not reuse the paper DOI as a data DOI.
7. Update the article's Data Availability Statement and arXiv entry using the
   actual accessible URLs. No software/data DOI has been registered for this release.

`dist/ST-GRAPE-public-v1.0.0.zip` is the complete shareable project including data.
The smaller wheel contains code only. `dist/SHA256SUMS` identifies the prepared
release artifacts. If any release file is changed, rebuild the archive/checksums.

Generated plots and local test outputs are not release inputs and are excluded
from the archives and Git. The `data/reference_figures/` files preserve the
original manuscript artwork; regenerated figures are computational reproductions,
not promises of byte-identical PDF metadata or layout across Matplotlib versions.
