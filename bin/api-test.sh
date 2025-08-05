#!/bin/bash

read -p "Enter Python version (e.g., 3.8, 3.9, 3.10, 3.11, 3.12): " version

py -$version test/api-test.py