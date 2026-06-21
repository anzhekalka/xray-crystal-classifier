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
"""C_SYSTEMS = ["cubic"]"""
C_SYSTEMS = [ "tetragonal",
"orthorhombic",
"hexagonal",
"trigonal",
"monoclinic",
"triclinic"]

def save_pattern_as_image(pattern, filepath):
    """
    converts XRD pattern into a 2D reciprocal space pixel strip.
    Intensity values are mapped directly to pixel brightness (0-255)
    along a 1D strip, then duplicated vertically to fill 224x224.
    This preserves exact peak position through CNN downsampling,
    unlike a sparse line plot where peaks can be lost in pooling.
    """
    x_peaks = np.array(pattern.x)
    y_peaks = np.array(pattern.y)

    # build a continuous intensity profile across the full angle range
    width = 224
    angle_range = np.linspace(0, 90, width)
    intensity = np.zeros(width)

    sigma = 0.3  # peak width in degrees
    for peak_angle, peak_intensity in zip(x_peaks, y_peaks):
        intensity += peak_intensity * np.exp(
            -0.5 * ((angle_range - peak_angle) / sigma) ** 2
        )

    # normalize to 0-255 pixel brightness range
    if intensity.max() > 0:
        intensity = intensity / intensity.max()
    pixel_strip = (intensity * 255).astype(np.uint8)

    # duplicate the 1D strip vertically to create a 224x224 image
    image_array = np.tile(pixel_strip, (224, 1))  # shape: (224, 224)

    # save as a single-channel grayscale PNG
    img = Image.fromarray(image_array, mode='L')
    img.save(filepath)


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
        
