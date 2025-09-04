# Development Environment

!!! info "`xwr` uses `uv`"

    While `pyproject.toml` reflects known limitations on dependency versions, we use `uv` to manage a lock file used for development.

    On a fresh linux install, you will need to [install `uv`](https://docs.astral.sh/uv/getting-started/installation/):
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    ```

## Setup

Set up fresh development machine:
```sh
sudo apt-get install -y openssh-server git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

ssh-keygen
# (add ssh key or add deploy key to xwr)
git clone git@github.com:WiseLabCMU/xwr.git
```

Set up development environment:
```sh
uv sync --all-extras --frozen
uv run pre-commit install
```

!!! warning

    This will install all backends (numpy, pytorch, and jax), which is necessary for static type checking and backend-specific tests. If you are only working on the interface, you can skip `--all-extras`.

## Tests

Run tests:
```sh
uv run --all-extras pytest -ra --cov --cov-report=html --cov-report=term -- tests
```

!!! tip

    HTML-format code coverage is saved to `./htmlcov`; view these with `cd htmlcov; python -m http.server 8001`.

Run tests (including data capture):
```sh
export XWR_DEVICE=AWR1843
uv run --all-extras pytest -ra --cov --cov-report=html --cov-report=term -- tests
```

- A radar and capture card should be configured and connected to the test computer.
- Set `XWR_DEVICE` to the name of the radar (i.e., a class name in [`xwr.radar`][xwr.radar]).
- If `XWR_DEVICE` is not set, all capture-related tests will be skipped.

!!! info
    
    The tests are also run by our [pre-commit](https://pre-commit.com/) hooks, which you can manually trigger with `uv run pre-commit run`; these hooks (`ruff + pyright + pytest`) mirror the CI.

## Docs

Build docs for development:
```sh
uv run --extra docs mkdocs serve
```

!!! info

    The [documentation site](https://radarml.github.io/xwr) is automatically built and deployed via GitHub Actions when a PR is merged.
