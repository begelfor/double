import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/evg/.ssh/resonant.json"
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
vertexai.init(project="resonant-feat-256420", location="us-central1")
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")


def generate_image_from_text(prompt: str, output_file: str, number_of_images: int = 1) -> str | list[str]:
    images = model.generate_images(
        prompt=prompt,
        # Optional parameters
        number_of_images=number_of_images,
        language="en",
        # You can't use a seed value and watermark at the same time.
        aspect_ratio="1:1",
        safety_filter_level="block_some",
        person_generation="allow_adult",
    )
    if number_of_images == 1:
        path = output_file.with_suffix(".jpg")
        images[0].save(location=path, include_generation_parameters=False)
        return path
    else:
        paths = []
        for i, image in enumerate(images):
            path = output_file.with_suffix(f"_{i}.jpg")
            image.save(location=path, include_generation_parameters=False)
            paths.append(path)
        return paths
    