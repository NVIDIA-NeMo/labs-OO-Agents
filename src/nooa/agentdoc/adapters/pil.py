# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for Pillow (``PIL``): curated doc() views for Image and ImageDraw.

Import to register:

    import nooa.agentdoc.adapters.pil

Without it, ``doc(PIL.Image.Image)`` is ~900 lines and ``doc(PIL.Image)`` ~600. With
it, they show the key functions and methods with the signatures of the installed
Pillow (read with ``inspect.signature``), plus idioms and pitfalls. Doc extraction
never calls Pillow functions.
"""

from PIL import Image, ImageDraw

from nooa.agentdoc import spec
from nooa.agentdoc.adapters._libdoc import lib_callable
from nooa.agentdoc.ext import FieldInfo, ModuleInfo, TypeInfo

# Executable idioms (tests run them with np, Image, ImageDraw and a writable `path`).
IMAGE_IDIOMS = """\
img = Image.new("RGB", (64, 32), (255, 0, 0))    # size is (width, height)
img.save(path)                                   # format from the extension (.png)
img = Image.open(path).convert("RGB")            # open is lazy; convert loads pixels
w, h = img.size; assert (w, h) == (64, 32) and img.mode == "RGB"
arr = np.asarray(img)                            # (height, width, 3) uint8, read-only
assert arr.shape == (32, 64, 3) and arr.dtype == np.uint8
out = Image.fromarray(np.ascontiguousarray(arr[:, ::-1]))   # array -> image (mode from dtype/shape)
small = img.resize((32, 16), Image.Resampling.LANCZOS)      # returns a new image
tile = img.crop((0, 0, 16, 16))                  # box = (left, upper, right, lower), right/lower exclusive
assert img.getpixel((0, 0)) == (255, 0, 0)       # xy = (x, y) = (column, row)
draw = ImageDraw.Draw(img); draw.rectangle((4, 4, 20, 20), outline=(0, 255, 0), width=2)"""

_IMAGE_PITFALLS = (
    "Pitfalls:\n"
    "- Sizes and coordinates are (x, y) = (width, height); NumPy arrays are (row, col) =\n"
    "  (height, width): np.asarray(img).shape == (h, w, channels).\n"
    "- Methods return new images (resize, crop, convert, rotate); paste, putpixel,\n"
    "  thumbnail and ImageDraw edit in place.\n"
    "- Image.open is lazy and keeps the file open: use `with Image.open(p) as im:` or\n"
    "  im.load()/convert() before the file changes. Save JPEG from 'RGB', not 'RGBA'.\n"
    "- Image.fromarray needs uint8 (or a matching dtype) and a C-contiguous array;\n"
    "  float arrays in [0, 1] must be scaled: (a * 255).clip(0, 255).astype(np.uint8)."
)

_MODES = (
    "Modes: '1' (bit), 'L' (8-bit gray), 'P' (palette), 'RGB', 'RGBA', 'CMYK', 'I' (int32),\n"
    "'F' (float32), 'I;16' (16-bit gray). Convert with img.convert('RGB')."
)


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


@spec.define_doc(Image)
def _image_module_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="PIL.Image",
        docstring=(
            "Pillow images. `from PIL import Image` (then Image.open, Image.Image).\n"
            "print(doc(Image.Image)) for the image object; print(doc(ImageDraw)) to draw.\n"
            + _MODES
            + "\nResampling filters: Image.Resampling.NEAREST, BILINEAR, BICUBIC, LANCZOS.\n"
            "Flips/rotations: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT / ROTATE_90 ...)."
        ),
        functions=[
            lib_callable(
                Image.open,
                "Open an image file (lazy: pixels load on first use). Raises\n"
                "FileNotFoundError or PIL.UnidentifiedImageError.",
            ),
            lib_callable(Image.new, "New image; color is a number, tuple or name ('white')."),
            lib_callable(
                Image.fromarray,
                "Image from a NumPy array: uint8 (H, W) -> 'L', (H, W, 3) -> 'RGB',\n"
                "(H, W, 4) -> 'RGBA'. Back to NumPy: np.asarray(img).",
            ),
            lib_callable(Image.frombytes, "Image from raw pixel bytes."),
            lib_callable(Image.merge, "Combine single-band images into one (e.g. 'RGB')."),
            lib_callable(Image.blend, "im1 * (1 - alpha) + im2 * alpha (same mode and size)."),
            lib_callable(Image.alpha_composite, "Composite RGBA im2 over RGBA im1."),
        ],
        classes=[
            ("Image", "the image object: size, mode, convert, resize, crop, save"),
            ("Resampling", "resize/rotate filters: NEAREST, BILINEAR, BICUBIC, LANCZOS"),
            ("Transpose", "FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90/180/270"),
        ],
    )


_I = Image.Image


@spec.define_doc(Image.Image)
def _image_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="Image",
        base="PIL.Image.Image",
        fields=[
            FieldInfo(name="size", type="tuple[int, int]", description="(width, height)."),
            FieldInfo(name="width", type="int"),
            FieldInfo(name="height", type="int"),
            FieldInfo(name="mode", type="str", description="'RGB', 'RGBA', 'L', ..."),
            FieldInfo(
                name="format", type="str | None", description="'PNG', 'JPEG'... for opened files."
            ),
            FieldInfo(name="info", type="dict", description="Format metadata (dpi, exif, ...)."),
        ],
        methods=[
            lib_callable(_I.convert, "Copy in another mode, e.g. convert('RGB'), convert('L')."),
            lib_callable(
                _I.resize,
                "Resized copy; size=(width, height). resample=Image.Resampling.LANCZOS etc.",
            ),
            lib_callable(
                _I.crop, "Copy of box=(left, upper, right, lower); right/lower exclusive."
            ),
            lib_callable(_I.save, "Write to a path or file; format from the extension if omitted."),
            lib_callable(_I.getpixel, "Pixel at (x, y): a tuple for RGB(A), a number for L."),
            lib_callable(_I.putpixel, "Set one pixel in place (slow; use NumPy for many)."),
            lib_callable(_I.copy, "Independent copy."),
            lib_callable(_I.paste, "Paste an image/color into this one in place (mask optional)."),
            lib_callable(
                _I.rotate, "Rotated copy, degrees counter-clockwise; expand=True grows canvas."
            ),
            lib_callable(_I.transpose, "Flip/rotate by 90s: Image.Transpose.FLIP_LEFT_RIGHT, ..."),
            lib_callable(_I.thumbnail, "Shrink in place to fit size, keeping the aspect ratio."),
            lib_callable(_I.split, "One single-band image per channel."),
            lib_callable(_I.getchannel, "One band, e.g. getchannel('A')."),
            lib_callable(_I.getbbox, "Bounding box of non-zero pixels, or None if all zero."),
            lib_callable(_I.filter, "Filtered copy, e.g. filter(ImageFilter.GaussianBlur(2))."),
            lib_callable(_I.tobytes, "Raw pixel bytes."),
            lib_callable(_I.load, "Load pixels now (closes lazily opened files' pending read)."),
            lib_callable(_I.close, "Release the file handle (or use `with Image.open(p) as im:`)."),
        ],
        docstring=(
            "A Pillow image. Get one from Image.open / Image.new / Image.fromarray.\n\n"
            "Idioms:\n" + _indent(IMAGE_IDIOMS) + "\n\n" + _IMAGE_PITFALLS + "\n\n" + _MODES
        ),
    )


_D = ImageDraw.ImageDraw


@spec.define_doc(ImageDraw)
def _imagedraw_module_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="PIL.ImageDraw",
        docstring=(
            "2-D drawing on a PIL image, in place:\n"
            "    draw = ImageDraw.Draw(img)\n"
            "    draw.line([(0, 0), (10, 10)], fill=(255, 0, 0), width=2)\n"
            "    draw.rectangle((x0, y0, x1, y1), fill=None, outline='white')   # x1/y1 inclusive\n"
            "    draw.text((x, y), 'label', fill='black')                       # default font\n"
            "Colors: a tuple matching img.mode, a number for 'L', or a name ('red').\n"
            "print(doc(ImageDraw.ImageDraw)) for all drawing methods."
        ),
        functions=[lib_callable(ImageDraw.Draw, "Drawing context for img.")],
        classes=[("ImageDraw", "drawing methods: line, rectangle, ellipse, polygon, text")],
    )


@spec.define_doc(ImageDraw.ImageDraw)
def _imagedraw_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="ImageDraw",
        base="PIL.ImageDraw.ImageDraw",
        fields=[],
        methods=[
            lib_callable(_D.line, "Polyline through xy = [(x, y), ...] or [x0, y0, x1, y1, ...]."),
            lib_callable(_D.rectangle, "Box xy = (x0, y0, x1, y1), corners inclusive."),
            lib_callable(_D.ellipse, "Ellipse inside the box xy = (x0, y0, x1, y1)."),
            lib_callable(_D.polygon, "Closed polygon through xy = [(x, y), ...]."),
            lib_callable(_D.point, "Single pixels at xy."),
            lib_callable(
                _D.text,
                "Draw text with its top-left at xy (anchor='la'); font from ImageFont.",
                annotate=False,
            ),
            lib_callable(
                _D.textbbox, "Bounding box (x0, y0, x1, y1) the text would cover.", annotate=False
            ),
        ],
        docstring="Create with ImageDraw.Draw(img); every method draws on img in place.",
    )
