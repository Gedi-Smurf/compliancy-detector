from PIL import Image, ImageOps


def load_image_rgb(path: str) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)

    # palette image with transparency -> go RGBA first
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    else:
        image = image.convert("RGB")

    return image