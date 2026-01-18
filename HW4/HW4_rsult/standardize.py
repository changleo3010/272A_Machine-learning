#!/usr/bin/env python3
# standardize.py
"""
Read diabetes.csv in the current directory and standardize the BMI column using z-score.
The standardized BMI values replace the original BMI values in the same file.
Other columns and the file name are preserved.
"""
import csv
import math


def compute_statistics(values):
    # values: list of floats
    if not values:
        return None, None
    mean = sum(values) / len(values)
    # sample standard deviation
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    stdev = math.sqrt(var)
    return mean, stdev


def main():
    path = "diabetes.csv"  # read/write in the current working directory

    # Read CSV and locate BMI column
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("diabetes.csv is empty.")
            return
        try:
            bmi_idx = header.index("BMI")
        except ValueError:
            print("BMI column not found in diabetes.csv header.")
            return
        rows = [row for row in reader]

    # Collect numeric BMI values for statistics
    bmi_values = []
    for r in rows:
        if bmi_idx < len(r):
            v = r[bmi_idx]
        else:
            v = ""
        try:
            bmi_values.append(float(v))
        except Exception:
            # Non-numeric values are ignored for statistics
            pass

    mean, stdev = compute_statistics(bmi_values)
    if mean is None or stdev is None:
        print("No BMI values available for standardization.")
        return

    # If no variation, set all numeric BMI values to 0.0 to avoid divide-by-zero
    if stdev == 0:
        for r in rows:
            if bmi_idx < len(r):
                v = r[bmi_idx]
                try:
                    float(v)
                    r[bmi_idx] = "0.0"
                except Exception:
                    pass
    else:
        for r in rows:
            if bmi_idx < len(r):
                v = r[bmi_idx]
                try:
                    f = float(v)
                    z = (f - mean) / stdev
                    # Keep a reasonable precision
                    r[bmi_idx] = f"{z:.6f}"
                except Exception:
                    pass

    # Write back to the same file, preserving header and order
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
