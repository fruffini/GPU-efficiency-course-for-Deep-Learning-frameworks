#!/usr/bin/env python3
"""
Generate sample NIfTI medical images for benchmarking
"""

import numpy as np
import nibabel as nib
import os
from pathlib import Path
import argparse


def generate_sample_nifti(output_dir, num_samples=20, fixed_shape=(128, 128, 128)):
    """Generate synthetic medical images with FIXED shape"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} sample NIfTI files...")
    print(f"Fixed shape: {fixed_shape}")

    for i in range(num_samples):
        # Create realistic-looking medical image with FIXED shape
        shape = fixed_shape  # Always use same shape

        # Background
        data = np.random.randn(*shape).astype(np.float32) * 50 + 100

        # Add some "organs" (spheres and ellipsoids)
        center = np.array(shape) // 2
        for _ in range(5):
            organ_center = center + np.random.randint(-30, 30, 3)
            radius = np.random.randint(10, 25)

            # Create sphere
            coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
            distance = np.sqrt(
                (coords[0] - organ_center[0])**2 +
                (coords[1] - organ_center[1])**2 +
                (coords[2] - organ_center[2])**2
            )
            mask = distance < radius
            data[mask] = np.random.randint(150, 250)

        # Add some noise
        data += np.random.randn(*shape) * 10

        # Clip to realistic HU range
        data = np.clip(data, -1000, 1000)

        # Create NIfTI image with consistent affine
        affine = np.diag([1.5, 1.5, 2.0, 1.0])  # Consistent spacing
        img = nib.Nifti1Image(data, affine=affine)

        # Save
        filename = output_dir / f"sample_{i:03d}.nii.gz"
        nib.save(img, filename)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_samples}...")

    print(f"✓ Generated {num_samples} samples in {output_dir}")
    print(f"  Total size: {sum(f.stat().st_size for f in output_dir.glob('*.nii.gz')) / 1024**2:.1f} MB")

    # Verify all files have same shape
    print("\nVerifying shapes...")
    shapes = []
    for f in sorted(output_dir.glob("*.nii.gz")):
        img = nib.load(f)
        shapes.append(img.shape)

    unique_shapes = set(shapes)
    if len(unique_shapes) == 1:
        print(f"✓ All files have consistent shape: {list(unique_shapes)[0]}")
    else:
        print(f"⚠ Warning: Found {len(unique_shapes)} different shapes: {unique_shapes}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample NIfTI files for benchmarking")
    parser.add_argument("--output-dir", "-o", default="chapter2_data_efficiency/data/nifti_samples",
                        help="Directory to write sample NIfTI files")
    parser.add_argument("--num-samples", "-n", type=int, default=50,
                        help="Number of sample NIfTI files to generate")
    parser.add_argument("--shape", "-s", type=int, nargs=3, metavar=("X", "Y", "Z"),
                        default=(128, 128, 128),
                        help="Volume shape as three integers, e.g. -s 128 128 128")
    args = parser.parse_args()

    generate_sample_nifti(args.output_dir, num_samples=args.num_samples, fixed_shape=tuple(args.shape))


if __name__ == "__main__":
    main()
