from PIL import Image
from torchvision import transforms


def img_to_tensor(img: str = "pali.png", img_size: int = 256):
    """
    Convert an image to a tensor.

    Args:
        img (str): The path to the image file. Default is "pali.png".
        img_size (int): The desired size of the image. Default is 256.

    Returns:
        torch.Tensor: The image converted to a tensor.

    """
    # Load image
    image = Image.open(img)

    # Define a transforms to convert the image to a tensor and apply preprocessing
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((img_size, img_size)),  # Resize the image to 256x256
            transforms.ToTensor(),  # Convert the image to a tensor,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize the pixel values
        ]
    )

    # apply transforms to the image
    x = transform(image)

    # Add batch dimension
    x = x.unsqueeze(0)

    return x
