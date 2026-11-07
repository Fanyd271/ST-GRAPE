# Release verification

## Completed locally

- Independent checkout and clean-environment package installation.
- 121 tests passed, including numerical gradients and paper data.
- Ruff lint, formatting, Pyright and CFF schema checks passed.
- 974 data file hashes verified; every NPZ file matches its source bytes.
- 76 printed table entries and four stress fidelity summaries reproduced.
- All five numerical paper figures and reply-only comparisons regenerated.
- Representative numerical figures visually reviewed.
- Wheel installed and run outside the source checkout.
- Original project's 3927 substantive-file hashes and Git state unchanged.
- No private absolute paths or credential patterns found in released text files.
- MIT code and CC BY 4.0 data licenses included.

## Public distribution

- Public repository: https://github.com/Fanyd271/ST-GRAPE.
- CI results: https://github.com/Fanyd271/ST-GRAPE/actions.
- Versioned downloads: https://github.com/Fanyd271/ST-GRAPE/releases.
- Long-term archiving and a separate software/data DOI remain optional follow-up work.
- The manuscript DAS can now reference the public repository; it is not edited by this release.

The historical development checkout had uncommitted changes. The manifest's
source commit identifies the extraction checkout, not a claim that every historical
experiment was run at that exact commit. Original numeric archives and their hashes
are retained. Full-size timing/optimization runs were not repeated. See
`verification.json` for the precise tested environment and limits.
