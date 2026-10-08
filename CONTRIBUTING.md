# Contributing

## Running the tests

```bash
uv run pytest tests/ -q
```

## Ad-hoc scripts in a worktree import the wrong branch

The venv is shared across worktrees, and the editable install inside it is a `.pth` file naming
**one** absolute path: the main checkout's `src`. So a plain `python some_script.py` launched from a
worktree imports `every_query` from the *main checkout*, which is on a different branch — no error,
no warning, just the wrong code.

`pyproject.toml` sets `pythonpath = ["src"]`, which fixes this **for pytest and only for pytest**. A
measurement or analysis script gets no such protection. This has already cost a real result: a
blast-radius measurement run this way compared the pre-fix code against itself and reported
"0 rows changed".

Two habits avoid it:

1. Put this checkout's `src` at the **front** of `PYTHONPATH` before running anything ad hoc:

    ```bash
    PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python path/to/script.py
    ```

2. Assert it, rather than trusting it. The deleted `scripts/experiments/_common.sh` guard was worth
    copying into any new driver:

    ```bash
    .venv/bin/python -c 'import every_query, pathlib; print(pathlib.Path(every_query.__file__).parent.parent)'
    ```

    (There is no bare `python` on the usual dev box — use `uv run python` or the venv's interpreter
    directly.)

    If that is not `<this checkout>/src`, stop — do not measure.
