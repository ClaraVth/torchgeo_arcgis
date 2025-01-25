import rasterio

def cut_tiff(input_path, output_path, cut_percentage=0.2):
    """
    Cuts a percentage of the height of a TIFF image.

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the cropped TIFF file.
        cut_percentage (float): Percentage of height to cut (e.g., 0.2 for 20%).
    """
    with rasterio.open(input_path) as src:
        # Read metadata and dimensions
        width, height = src.width, src.height
        transform = src.transform

        # Calculate new dimensions
        cut_pixels = int(height * cut_percentage)
        new_height = height - cut_pixels

        # Update transform to reflect the cropped region
        new_transform = transform * rasterio.Affine.translation(0, cut_pixels)

        # Read the data from the cropped region
        cropped_data = src.read(window=rasterio.windows.Window(0, cut_pixels, width, new_height))

        # Update metadata for the cropped image
        new_meta = src.meta
        new_meta.update({
            "height": new_height,
            "transform": new_transform
        })

        # Write the cropped image to a new file
        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(cropped_data)

def cut_tiff_remainder(input_path, output_path, cut_percentage=0.2):
    """
    Extracts a percentage of the height of a TIFF image (e.g., the top or bottom part).

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the cropped TIFF file.
        cut_percentage (float): Percentage of height to extract (e.g., 0.2 for 20%).
    """
    with rasterio.open(input_path) as src:
        # Read metadata and dimensions
        width, height = src.width, src.height
        transform = src.transform

        # Calculate dimensions for the remaining 20%
        cut_pixels = int(height * cut_percentage)

        # Define window for the "removed" part (e.g., top 20%)
        remainder_window = rasterio.windows.Window(0, 0, width, cut_pixels)

        # Update transform to reflect the cropped region
        new_transform = transform

        # Read the data from the cropped region
        cropped_data = src.read(window=remainder_window)

        # Update metadata for the cropped image
        new_meta = src.meta
        new_meta.update({
            "height": cut_pixels,
            "transform": new_transform
        })

        # Write the cropped image to a new file
        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(cropped_data)

# Input and output file paths
input_tiff = "data/multispectral_image.TIF"
output_tiff = "data/training_image_cropped.tif"

# Cut 20% from the top
#cut_tiff(input_tiff, output_tiff, cut_percentage=0.2)
output_tiff = "data/test_image_cropped.tif"
cut_tiff_remainder(input_tiff, output_tiff, cut_percentage=0.2)

print(f"Cropped TIFF saved to: {output_tiff}")