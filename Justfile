set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set shell := ["bash", "-cu"]

PYTHON := "3.13"
EXCLUDE := "scripts"

# Cross-platform project name extraction (works on macOS and Windows)
PROJECT_NAME := `python -c "import os; print(os.path.basename(os.getcwd()))"`

# Show available commands
help:
    @just --list

# Setup the current development environment
setup:
    @echo "Setting up the development environment..."
    @uv sync --python={{ PYTHON }}

# Build the project, useful for checking that packaging is correct
build:
    @echo "Building the project..."
    @uv build

# Run quality assurance checks
qa:
    @uv run --python={{ PYTHON }} ruff format
    @uv run --python={{ PYTHON }} ruff check --fix
    @uv run --python={{ PYTHON }} ruff check --select I --fix
    @uv run --python={{ PYTHON }} mypy .
    @uv run --python={{ PYTHON }} deptry .

# Run tests without coverage
test:
    @echo "Running tests..."
    @uv run --python={{ PYTHON }} pytest #--disable-warnings tests/

# Run coverage checks, and generate xml reports
coverage:
    @echo "Running tests with coverage..."
    @uv run --python={{ PYTHON }} pytest --disable-warnings --cov=src tests/
    @uv run --python={{ PYTHON }} coverage report -m
    @uv run --python={{ PYTHON }} coverage xml
