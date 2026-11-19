"""Download the two task datasets and write them to data/raw/ as CSV.

Run this once anywhere with open network access (Colab, a laptop, Hopper):

    pip install PyTDC
    python fetch_data.py

It writes data/raw/caco2_wang.csv and data/raw/herg.csv, each with columns
Drug (SMILES) and Y (label). Those two files are all the rest of the pipeline
needs; after that, run:

    python src/run_baseline.py --prefer-cache
"""

import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

TASKS = [
    ("caco2_wang", "Caco2_Wang", "ADME"),
    ("cyp3a4", "CYP3A4_Veith", "ADME"),
    ("herg", "hERG", "Tox"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, tdc_name, group in TASKS:
        if group == "ADME":
            from tdc.single_pred import ADME as Loader
        else:
            from tdc.single_pred import Tox as Loader

        df = Loader(name=tdc_name, path=OUT_DIR).get_data()
        df = df[["Drug", "Y"]].dropna().reset_index(drop=True)
        path = os.path.join(OUT_DIR, f"{key}.csv")
        df.to_csv(path, index=False)
        print(f"{key}: {len(df)} molecules -> {path}")
        print(f"  label range: {df['Y'].min():.3f} to {df['Y'].max():.3f}")


if __name__ == "__main__":
    main()
