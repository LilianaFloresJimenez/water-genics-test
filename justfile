[private]
default:
    just --list --unsorted

# Installs pip environment, dependencies from requirements.txt and installs this project (package) in developer mode
setup: install-virtual-environment install-dependencies
    pip --require-virtualenv install -e .

# Installs python virtual environment
install-virtual-environment:
    echo $OSTYPE
    virtualenv .venv
    . .venv/bin/activate

# Installs dependencies from requirements.txt
install-dependencies:
    pip --require-virtualenv install -r requirements.txt

# Runs tests with simple text code coverage overview report
test:
    pytest tests/ -v --cov=src

# Runs tests with detailed html code coverage report
test-with-html-coverage:
    pytest tests/ -v --cov=src --cov-report=html
    open htmlcov/index.html

# Remove all mlflow data
mlflow-clear:
    rm -Rf mlruns/*
    rm -Rf mlruns/.trash

# Open mlflow ui in browser
mlflow-show-ui:
    open http://127.0.0.1:5000
    mlflow ui