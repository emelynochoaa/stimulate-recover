#!/bin/bash

# Run test suite 
python3 -m unittest discover -s test -p "test_*.py"

echo "All tests executed."
