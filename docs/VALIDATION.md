# Development validation gates

Application source and structure are unchanged. `pyproject.toml` declares the `edalens`
distribution (flat layout, setuptools) and configures tools; there is no PyPI release
workflow, and the gates below never upload anything.

## Set up and run

Use Python 3.10 (the Azure baseline). Check the interpreter version before creating
the environment. Windows PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-validation.lock
.\.venv\Scripts\python.exe scripts/validate.py
```

On POSIX, create the environment with `python3.10 -m venv .venv` and substitute
`.venv/bin/python` for the Windows executable. No activation is necessary.

The validation command fails on the first failing gate and does not install or
update dependencies. Lock freshness needs network access (or a populated uv cache);
an unavailable dependency index fails the check rather than silently skipping it.

Gates:

1. Python 3.10 and active-source syntax compatibility.
2. Installed dependency consistency (`pip check`) and installed versions matching the lock.
3. Hash-locked dependency files match their input requirements, using pinned uv.
4. Ruff correctness checks (`E9`, `F63`, `F7`, `F82`) with an exact existing-error baseline.
5. Formatting of the launcher, validation scripts, and tests.
6. Regression tests for transformations: column, index, order, and dtype
   preservation (including imputation of nullable integer, datetime, categorical,
   all-missing, and empty data), explicit `OperationError` failures, agreement of
   direct execution and replay, atomic session updates, undo and reset, a
   read-only Visualization page, deterministic sampling, and initial Home rendering; launcher tests for
   localhost binding, occupied/invalid ports, startup failure and timeout, foreign
   servers on the port, stopping only its own process, optional browser opening,
   the CLI, and one real Streamlit start/stop.
7. Tests that both Azure deployments depend on validation and that the deployment
   archive contains only runtime sources plus hashed runtime requirements.
8. Packaging (`scripts/check_package.py`): `validate-pyproject`; sdist build and a
   wheel built from that sdist, using the locked setuptools without build isolation;
   `check-wheel-contents` with `edalens` as the only top-level entry
   (W201/W202 replace W009) and source-tree comparison
   (W101/W102); `twine check --strict` on both artifacts; `pyroma --min 8` on the
   sdist; and installation of the wheel into a temporary target that is imported and
   rendered with Streamlit's `AppTest` from outside the checkout, whose `edalens`
   console-script entry point is declared and whose launcher starts the installed
   app, serves it, and stops it. Tests also keep
   `[project.dependencies]` equal to `requirements-runtime.in` and prove the gate
   stops on invalid metadata.

Each subprocess has a five-minute limit; CI has a twenty-minute job limit.
New tests in `tests/` are automatically collected. Add a reproducing regression
test before fixing an application defect, then run the same validation command.

## Existing lint debt

`scripts/lint-baseline.json` records one existing `F821` finding in
`views/time_series.py`: the call to undefined `run_prophet_changeppoints`.
Matching uses path, rule, message, source text, and occurrence count, not a broad
file exclusion. New or duplicated findings fail. Once fixed, the stale baseline
entry also fails and must be removed. Do not expand the baseline to hide regressions.

The gate deliberately does not impose full formatting, full linting, or typing on
legacy application code. Those require separate reviewed adoption work. Tests do
not yet cover every page, statistical output, or known defect.

## Dependencies

- `requirements.txt`: existing application inputs, unchanged.
- `requirements-runtime.in`: includes those inputs and declares `requests`, already
  imported by both loaders but previously only available transitively.
- `requirements-runtime.lock`: pinned, hashed runtime resolution.
- `requirements-validation.in` / `.lock`: runtime constraints plus validation tools.

Regenerate without intentionally upgrading existing resolutions:

```sh
python scripts/lock_dependencies.py
```

For an intentional dependency refresh, use `--upgrade`. Then review both lock
diffs, reinstall with `--require-hashes`, and run validation. `--check` never
changes the committed locks. The universal locks contain platform/Python markers,
but only Python 3.10 is currently validated; they do not claim broader support.

For application-only installation use:

```sh
python -m pip install --require-hashes -r requirements-runtime.lock
python -m streamlit run my_viz_plotly.py
```

The initial lock freezes a resolution of the previously unpinned inputs; it is not
a reconstruction of an existing Azure environment. Review and test it in staging
before production deployment. Hashes verify downloaded artifacts, not absence of
vulnerabilities. This setup does not include a vulnerability-audit gate yet.

## CI and deployment enforcement

`.github/workflows/validate.yml` runs on pull requests and pushes to `main` and is
reused by both deployment workflows. Failed validation prevents deployment. The
deploy jobs download the archive built in their own successful validation run;
they no longer zip the entire checkout independently.

`scripts/build_deployment.py` includes only `my_viz_plotly.py`, the `edalens/`
package sources, and runtime requirements under Azure's expected `requirements.txt`
name. No environment, development tools, legacy root copies, or secrets are added.
Archive ordering and timestamps are fixed. Extend this allowlist explicitly if
the application later gains local runtime assets. The Azure startup configuration
remains external to this repository.

To enforce merge protection, a repository administrator must enable a `main`
ruleset requiring pull requests and the **Python 3.10 validation** status check
(select the check emitted by the standalone **Validation** workflow after its
first run), block force pushes, and restrict bypass privileges. Require branches
to be up to date if using strict status checks. This remote setting has not been
configured by these file changes. Review changes to workflows, locks, tests, and
baselines as security-relevant code; they can change what the gate enforces.

Local commands are not an unbypassable security boundary. CI gates run after the
files are pushed; remote rulesets are what prevent merging without a successful
check. Nothing in this change publishes to PyPI or triggers an Azure deployment
until the workflows are pushed/dispatched.

## Packaging limits

The CI package gate installs the wheel with `--no-deps` into the validation
environment, so it does not prove resolution in an empty environment. Verify that
manually before any release: create a fresh Python 3.10 venv outside the checkout,
install `requirements-runtime.lock` with `--require-hashes`, install the wheel (and
separately the sdist) with `--no-deps`, then run `pip check`, the `edalens` command
(stop it with Ctrl+C), and `edalens.show()` / `.stop()` from Python.
The wheel uses the generic top-level names `views` and `utils`; a `src/` layout
with a single `edalens` import package would avoid collisions but is not done.
pyroma currently scores 8/10 (no author email by decision), exactly the threshold.
