import torch
from torchvision import io


def video_to_tensor(file_path):
    """
    Transforms a video file into a PyTorch tensor.

    Args:
        file_path (str): The path to the video file.

    Returns:
        video_tensor (torch.Tensor): A tensor representation of the video.
        audio_tensor (torch.Tensor): A tensor representation of the audio.
    """
    # Load the video file
    video_tensor, audio_tensor, info = io.read_video(file_path, pts_unit="sec")

    return video_tensor, audio_tensor


def video_to_tensor_vr(file_path):
    """
    Transforms a video file into a PyTorch tensor.

    Args:
        file_path (str): The path to the video file.

    Returns:
        video_tensor (torch.Tensor): A tensor representation of the video.
        audio_tensor (torch.Tensor): A tensor representation of the audio.
    """
    # Create a VideoReader object
    reader = io.VideoReader(file_path, "video")

    # Get the metadata of the video
    reader.get_metadata()

    # Set the current stream to the default video stream
    reader.set_current_stream("video:0")

    # Initialize a list to hold the video frames
    frames = []

    # Read the video frames one by one
    for frame in reader:
        frames.append(frame["data"])

    # Convert the list of frames into a tensor
    video_tensor = torch.stack(frames)

    # Since the VideoReader does not support audio, we return None for the audio tensor
    audio_tensor = None

    return video_tensor, audio_tensor
