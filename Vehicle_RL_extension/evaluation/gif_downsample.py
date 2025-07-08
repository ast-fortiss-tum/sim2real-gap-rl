from PIL import Image, ImageSequence

"""
Author: Cristian Cubides
This script down-samples an existing GIF by skipping frames, resizing, and quantizing.
It uses the Python Imaging Library (PIL) to handle GIF images.
The output GIF will have fewer frames, reduced resolution, and a limited color palette.
This is useful for reducing file size while maintaining visual quality.
"""

def downsample_gif(
    in_path: str,
    out_path: str,
    skip: int = 2,
    scale: float = 0.5,
    colors: int = 64
):
    """
    Downsample an existing GIF by skipping frames, resizing, and quantizing.
    
    Parameters:
      in_path   – source GIF filename
      out_path  – output GIF filename
      skip      – keep every n-th frame (e.g. skip=2 → 50% frames)
      scale     – resize factor (e.g. 0.5 → half width & height)
      colors    – max palette size (≤256)
    """
    im = Image.open(in_path)
    frames = []
    
    # iterate, skip, resize & quantize
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i % skip != 0:
            continue
        # ensure RGBA to preserve transparency, then resize
        frame = frame.convert("RGBA")
        w, h = frame.size
        frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        # convert to P-palette with adaptive palette
        p = frame.convert('P', palette=Image.ADAPTIVE, colors=colors)
        frames.append(p)

    # save optimized GIF
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        optimize=True
    )

# Example usage:
downsample_gif(
    in_path="./gifs/<name>.gif",
    out_path="./gifs/<name>_downsampled.gif",
    skip=10,      # keep every 3rd frame
    scale=1.0,   # half the resolution
    colors=64    # limit palette
)
