"""
data_download.py - generates synthetic XRD diffraction pattern
images for all 7 crystal systems using pymatgen and the
Materials Project database.
"""
from typing_extensions import NotRequired #patch for Python 3.10 compatibility
import typing
typing.NotRequired = NotRequired

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from PIL import Image
import pymatgen.core as mg
from pymatgen.analysis.diffraction.xrd import XRDCalculator #takes atom positions and calculates where diffraction peaks appear mathematically

from mp_api.client import MPRester #the client that talks to Materials Project database
import os
from dotenv import load_dotenv #api 



#api loading
load_dotenv()
api_key = os.getenv("MATERIALS_PROJECT_API_KEY")

#defining 

C_SYSTEMS = ["tetragonal",
"orthorhombic",
"hexagonal",
"trigonal",
"monoclinic",
"triclinic"]

def save_pattern_as_image(pattern, filepath):
    """
    takes raw diffraction data (lists of numbers) and converts it into a PNG image file
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=56)

    x = pattern.x
    y = pattern.y

    y_max = max(y) if max(y) > 0 else 1 #normalizing intensities so tallest peak = 1.0
    y_norm = [val/y_max for val in y]

    for xi, yi in zip(x, y_norm):
        color = plt.cm.hot(yi)  # hot colormap: black→red→yellow→white
        ax.plot([xi, xi], [0, yi], color=color, linewidth=1.5)
    
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig(filepath, bbox_inches='tight',
                pad_inches=0, facecolor='black', dpi=56)
    plt.close(fig)


def generate_dataset():
    """
    generates and stores 100 XRD diffraction patterns 
    for each crystal system in data/crystals/{crystal}/
    """
    calculator = XRDCalculator(wavelength="CuKa")

    with MPRester(api_key) as mpr:
        for crystal in C_SYSTEMS:
            print(f"Downloading {crystal}...")

            docs = mpr.materials.summary.search( #Downloads 100 crystal structures
                crystal_system=crystal.capitalize(),
                fields=["structure", "symmetry"],
                num_chunks=5,
                chunk_size=50
            )
            structures = [doc.structure for doc in docs if doc.structure is not None]#only keeping the structure
            print(f"  found {len(structures)} structures")

            for i, structure in enumerate(structures):
                try:
                    pattern = calculator.get_pattern(structure) #returns peak positions and intensities
                    filepath = f"data/crystals/{crystal}/{crystal}_{i:03d}.png"
                    folder = f"data/crystals/{crystal}"
                    os.makedirs(folder, exist_ok=True)
                    save_pattern_as_image(pattern, filepath)
                except Exception as e:
                    print(f"  skipping structure {i}: {e}")

            print(f"  saved {len(os.listdir(f'data/crystals/{crystal}'))} images")


if __name__ == "__main__":
    generate_dataset()
    print("Dataset complete!")
        
