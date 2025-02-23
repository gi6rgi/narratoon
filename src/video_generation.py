import subprocess

from moviepy import AudioFileClip, ImageClip


def run_wav2lip(
    face_video: str, audio_path: str, output_path: str = "output.mp4"
) -> str:
    command = [
        "python",
        "inference.py",
        "--checkpoint_path",
        "checkpoints/wav2lip_gan.pth",
        "--face",
        face_video,
        "--audio",
        audio_path,
        "--outfile",
        output_path,
    ]
    subprocess.run(command, cwd="Wav2Lip", check=True)

    return output_path


def generate_video_from_photo(
    image_path: str, duration: float, output_video_path: str, fps: int = 24
) -> None:
    image_clip = ImageClip(image_path).with_duration(duration)
    image_clip.write_videofile(output_video_path, fps=fps, codec="libx264", audio=False)


def generate_video(
    image_path: str, audio_path: str, output_path: str = "output.mp4"
) -> str:
    with AudioFileClip(audio_path) as audio_clip:
        duration = audio_clip.duration

    static_character_video_path = "character_static.mp4"
    static_video_path = generate_video_from_photo(
        image_path=image_path,
        duration=duration,
        output_video_path=static_character_video_path,
    )
    return run_wav2lip(face_video=static_video_path, audio_path=audio_clip)
