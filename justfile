# Run all tests
test:
    uv run python -m unittest discover -s "tests" -t "." --verbose

# Run a single test file (e.g., just test-one test_graph)
test-one FILE:
    uv run python -m unittest tests.{{FILE}} -v

# Run type checking
typecheck:
    uv run mypy pyzx/ tests/

# Build HTML documentation
docs:
    uv run sphinx-build -M html doc doc/_build

# Build PDF documentation
docs-pdf:
    uv run sphinx-build -M latexpdf doc doc/_build

# Launch Jupyter notebook
notebook:
    uv run jupyter notebook
