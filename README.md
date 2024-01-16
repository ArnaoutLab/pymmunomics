# pymmunomics

## Installation instructions

```sh
pip install --upgrade pip
pip install "pymmunomics @ git+https://github.com/JasperBraun/pymmunomics"
```

for developers:
```sh
pip install --upgrade pip
git clone https://github.com/JasperBraun/pymmunomics
cd pymmunomics
pip install -e ".[dev]"
```

## Developers notes

***Do not make changes to `main` branch or `docs` branch without reading this:***

- Make changes in separate branch and when tests pass, merge into `main`.
- Every time `main` is updated, do the following:
  - merge `main` into `docs`
  - run sphinx (`cd docs` and `./build.sh`)
  - push changes in `docs` branch to github
  - Make sure to update the version in `pyproject.toml` accordingly

- **Do not merge `docs` into `main`**
- **Do not make changes in `docs` branch outside docs directory**

This set-up is the best way I know at the time of this writing to keep
the `docs` branch on a separate branch and not have it be part of the
user installation.
