# Installation of the Package

For development work on the project, it is recommended to follow the instructions in `CONTRIBUTING.md`.

## Using in External Project or Scripts

You can install the latest version of diffpssi via pip. For that just run 
```bash
$ pip install diffpssi
```
in your desired Python development environement.

Otherwise, you can also clone the repo and install it from as a local source with pip. For that just run
```bash
- PATH_TO_REPO $ git clone git@github.com:maxikoehler/diffpssi.git
- your_workspace $ pip install PATH_TO_REPO
```

## Modifying the Package (Developers Version)

The installation with this method is suggested if one wants to extend or modify the package. For version resolving and packaging, `poetry` is used.

Start with cloning the repository: 
```bash
$ git clone git@github.com:maxikoehler/diffpssi.git
```
After that, create a virtual environment, and activate it. Upgrade pip and install poetry.
```bash
$ python -m venv .venv
$ .venv/Source/activate  # for Windows
$ source .venv/bin/activate # for Linux / MacOS 

$ python -m pip install --upgrade pip
$ pip install poetry==2.1.4
```
When the project is initialized, install pre-commit hooks and checkout to the development branch.
```bash
$ pre-commit install
$ git checkout development
```
Further instructions to contribute are included in `CONTRIBUTING.md`.
