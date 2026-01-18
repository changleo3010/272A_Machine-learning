#!/usr/bin/env python3
# stand_check.py
"""
Read the (updated) diabetes.csv and compute the mean and standard deviation of the BMI column.
The script will try to detect BMI column by common names (BMI, BMIin, or any header containing bmi).
It prints: mean: <value> std: <value>
"""
import csv
import math
import os


def find_bmi_index(header):
    # Try common exact names first
    if "BMI" in header:
        return header.index("BMI")
    if "BMIin" in header:
        return header.index("BMIin")
    # Fallback: case-insensitive substring match for headers containing 'bmi'
    for i, col in enumerate(header):
        if isinstance(col, str) and "bmi" in col.lower():
            return i
    return None


def main():
    path = "diabetes.csv"  # relative to current working directory
    if not os.path.exists(path):
        print("diabetes.csv not found.")
        return

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("diabetes.csv is empty.")
            return

        bmi_idx = find_bmi_index(header)
        if bmi_idx is None:
            print("BMI column not found in diabetes.csv header.")
            return

        rows = [row for row in reader]

    # Collect numeric BMI values
    bmi_values = []
    for r in rows:
        if bmi_idx < len(r):
            v = r[bmi_idx]
        else:
            v = ""
        try:
            bmi_values.append(float(v))
        except Exception:
            # ignore non-numeric cells
            pass

    if not bmi_values:
        mean = 0.0
        std = 0.0
    else:
        mean = sum(bmi_values) / len(bmi_values)
        if len(bmi_values) < 2:
            std = 0.0
        else:
            var = sum((x - mean) ** 2 for x in bmi_values) / (len(bmi_values) - 1)
            std = math.sqrt(var)

    print("mean:", mean, "std:", std)


if __name__ == "__main__":
    main()
