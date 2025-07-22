## Installation Guide
Make sure virtualenv is installed on your system and pip is up to date.
It's possible to install the project with [just](https://github.com/casey/just):
```shell
    just setup
```
Alternatively, you can install it manually:
```shell
    # Installs python virtual environment
    pip install virtualenv
    virtualenv .venv
    . .venv/bin/activate
    pip install --upgrade pip
    # Installs dependencies from requirements.txt
    pip --require-virtualenv install -r requirements.txt
    # Installs this project (package) in developer mode
    pip --require-virtualenv install -e .
```