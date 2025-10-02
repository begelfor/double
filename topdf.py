from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import math
import os
from pathlib import Path

def create_image_grid_pdf(image_paths, output_pdf_path, images_per_row=2, images_per_column=3,
                          page_size=A4, margin=0.3 * inch):
    """
    Creates a PDF file with a grid of square JPEG images.

    Args:
        image_paths (list): A list of paths to the square JPEG image files.
        output_pdf_path (str): The path where the output PDF file will be saved.
        images_per_row (int): Number of images to place in each row.
        images_per_column (int): Number of images to place in each column.
        page_size (tuple): The size of the PDF page (e.g., letter, A4).
        margin (float): Margin around the images on the page.
    """

    if not image_paths:
        print("No image paths provided. PDF not created.")
        return

    c = canvas.Canvas(output_pdf_path, pagesize=page_size)
    page_width, page_height = page_size

    # Calculate available space for images
    available_width = page_width - (images_per_row+1) * margin
    available_height = page_height - (images_per_column+1)* margin

    # Assuming all images have the same dimensions, get them from the first image
    try:
        with Image.open(image_paths[0]) as img:
            original_img_width, original_img_height = img.size
            if original_img_width != original_img_height:
                print(f"Warning: Image '{image_paths[0]}' is not square. It will be fitted.")
    except IOError as e:
        print(f"Error opening image {image_paths[0]}: {e}")
        return

    # Calculate target size for each image in the grid
    # We want to maintain aspect ratio, and since they are square,
    # the target width and height will be the same.
    img_target_width = available_width / images_per_row
    img_target_height = available_height / images_per_column

    # Use the smaller dimension to ensure the image fits within its cell,
    # assuming square images, this will be the same.
    final_img_dim = min(img_target_width, img_target_height)

    images_on_page = images_per_row * images_per_column
    num_pages = math.ceil(len(image_paths) / images_on_page)

    for page_num in range(num_pages):
        start_index = page_num * images_on_page
        end_index = min(start_index + images_on_page, len(image_paths))
        current_page_images = image_paths[start_index:end_index]

        for i, img_path in enumerate(current_page_images):
            row = i // images_per_row
            col = i % images_per_row

            x_offset = margin + col * (final_img_dim+margin)
            y_offset = page_height - (row + 1) * (final_img_dim+margin) # Y-coordinates are from bottom-left

            try:
                c.drawImage(img_path, x_offset, y_offset, width=final_img_dim, height=final_img_dim)
            except Exception as e:
                print(f"Could not place image {img_path} on PDF: {e}")

        if page_num < num_pages - 1: # Don't add a new page after the last one
            c.showPage()

    c.save()
    print(f"PDF created successfully at: {output_pdf_path}")


def main():
    folder = Path('/home/evg/Data/double/cards_9a')
    image_paths = [str(f) for f in folder.glob('*.jpg')]
    create_image_grid_pdf(image_paths, str(folder/'double.pdf'), images_per_row=2, images_per_column=3,)

# --- Example Usage ---
if __name__ == "__main__":
    main()