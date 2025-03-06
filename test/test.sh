#!/bin/bash

# Run test suite
python3 -m unittest discover -s tests -p "test_*.py"

echo "All tests executed."

