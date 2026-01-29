from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:  # pragma: no cover
    # Pillow is required for meme generation. If missing, the app will fail at runtime.
    raise RuntimeError(
        "Pillow is required. Ensure Pillow is installed and present in requirements.txt"
    ) from e


def _env(name: str, default: str) -> str:
    """Internal helper to fetch environment variables with defaults."""
    return os.getenv(name, default)


def _safe_filename(original: str) -> str:
    """Create a safe filename preserving extension, for storing user uploads."""
    original = original or "upload"
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", original).strip("._")
    if not name:
        name = "upload"
    return name[:200]


def _ensure_dir(path: Path) -> None:
    """Internal helper to ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _resolve_asset_root() -> Path:
    """
    Resolve backend asset root directory for persisted files.

    We store assets under <repo>/data to avoid mixing with source code.
    """
    # src/api/main.py -> src/api -> src -> meme_backend
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / "data"


ASSET_ROOT = _resolve_asset_root()
TEMPLATES_DIR = ASSET_ROOT / "templates"
UPLOADS_DIR = ASSET_ROOT / "uploads"
GENERATED_DIR = ASSET_ROOT / "generated"

_ensure_dir(TEMPLATES_DIR)
_ensure_dir(UPLOADS_DIR)
_ensure_dir(GENERATED_DIR)


def _create_placeholder_templates_if_missing() -> None:
    """
    Create a small set of retro placeholder templates if the templates folder is empty.

    This keeps the app functional out-of-the-box without shipping binary assets.
    """
    existing = list(TEMPLATES_DIR.glob("*.png"))
    if existing:
        return

    presets = [
        ("retro-sunrise.png", ("RETRO", "SUNRISE"), (255, 0, 128), (0, 255, 255)),
        ("neon-grid.png", ("NEON", "GRID"), (0, 255, 180), (255, 64, 0)),
        ("vhs-glitch.png", ("VHS", "GLITCH"), (255, 255, 0), (128, 0, 255)),
    ]

    for filename, label, c1, c2 in presets:
        w, h = 900, 600
        img = Image.new("RGB", (w, h), (20, 18, 40))
        draw = ImageDraw.Draw(img)

        # Simple retro gradient banding.
        for y in range(h):
            t = y / max(h - 1, 1)
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Add a grid.
        grid_color = (255, 255, 255, 90)
        for x in range(0, w, 60):
            draw.line([(x, 0), (x, h)], fill=grid_color, width=2)
        for y in range(0, h, 60):
            draw.line([(0, y), (w, y)], fill=grid_color, width=2)

        # Center label
        font = _load_best_font(size=80)
        text = " ".join(label)
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.text(
            ((w - tw) / 2, (h - th) / 2),
            text,
            font=font,
            fill=(10, 10, 10),
            stroke_width=6,
            stroke_fill=(255, 255, 255),
        )

        img.save(TEMPLATES_DIR / filename, format="PNG")


def _load_best_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load a font for meme text.

    Uses DejaVuSans-Bold if available; otherwise falls back to PIL's default bitmap font.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    """Wrap text to fit within max_width."""
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    lines: List[str] = []
    current: List[str] = []

    for w in words:
        trial = (" ".join(current + [w])).strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width or not current:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]

    if current:
        lines.append(" ".join(current))
    return lines


def _draw_meme_text(
    img: Image.Image,
    top_text: str,
    bottom_text: str,
) -> Image.Image:
    """Draw top and bottom meme text with stroke for readability."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Dynamic sizing based on image size.
    base_size = max(18, int(h * 0.08))
    font = _load_best_font(size=base_size)

    margin = int(w * 0.04)
    max_text_width = w - 2 * margin

    def draw_block(text: str, at_top: bool) -> None:
        if not text.strip():
            return

        lines = _wrap_text(draw, text.upper(), font, max_text_width)
        if not lines:
            return

        # Compute line height.
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        line_height = max(line_heights) + int(base_size * 0.15)
        block_height = line_height * len(lines)

        y = margin if at_top else (h - margin - block_height)
        for idx, line in enumerate(lines):
            x = (w - line_widths[idx]) / 2
            draw.text(
                (x, y + idx * line_height),
                line,
                font=font,
                fill=(255, 255, 255),
                stroke_width=max(2, int(base_size * 0.10)),
                stroke_fill=(0, 0, 0),
            )

    draw_block(top_text, at_top=True)
    draw_block(bottom_text, at_top=False)
    return img


# Create placeholders early so /templates always has content.
_create_placeholder_templates_if_missing()


openapi_tags = [
    {
        "name": "Health",
        "description": "Service health & basic info.",
    },
    {
        "name": "Templates",
        "description": "Template gallery endpoints.",
    },
    {
        "name": "Uploads",
        "description": "Image upload endpoints (for custom images).",
    },
    {
        "name": "Memes",
        "description": "Meme generation endpoints.",
    },
]

app = FastAPI(
    title="Retro Meme Generator API",
    description=(
        "Backend for a web-based meme generator. Provides template gallery, image uploads, "
        "and server-side meme generation. Also serves images via /static."
    ),
    version="0.1.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For template use; lock down in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static serving for templates/uploads/generated
app.mount("/static/templates", StaticFiles(directory=str(TEMPLATES_DIR)), name="templates")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount(
    "/static/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated"
)


class TemplateItem(BaseModel):
    """Template metadata returned to the frontend."""

    id: str = Field(..., description="Template identifier (filename without extension).")
    name: str = Field(..., description="Human-friendly template name.")
    image_url: str = Field(..., description="Absolute or relative URL to the template image.")
    width: int = Field(..., description="Template width in pixels.")
    height: int = Field(..., description="Template height in pixels.")


class UploadResponse(BaseModel):
    """Response returned after uploading an image."""

    upload_id: str = Field(..., description="Server-side ID of the uploaded image.")
    image_url: str = Field(..., description="URL to access the uploaded image.")


class GenerateMemeRequest(BaseModel):
    """Request body for generating a meme."""

    template_id: Optional[str] = Field(
        default=None,
        description="Template id to use (mutually exclusive with upload_id).",
    )
    upload_id: Optional[str] = Field(
        default=None,
        description="Uploaded image id to use (mutually exclusive with template_id).",
    )
    top_text: str = Field(default="", description="Top caption text.")
    bottom_text: str = Field(default="", description="Bottom caption text.")


class GenerateMemeResponse(BaseModel):
    """Response returned after generating a meme."""

    meme_id: str = Field(..., description="Generated meme ID.")
    image_url: str = Field(..., description="URL to access the generated meme image.")


def _find_template_file(template_id: str) -> Optional[Path]:
    """Locate a template image by id."""
    if not template_id:
        return None
    # Expect template_id corresponds to filename stem.
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = TEMPLATES_DIR / f"{template_id}{ext}"
        if cand.exists():
            return cand
    return None


def _find_upload_file(upload_id: str) -> Optional[Path]:
    """Locate an uploaded image by upload_id."""
    if not upload_id:
        return None
    # upload files are saved as {id}_{safeName}
    matches = list(UPLOADS_DIR.glob(f"{upload_id}_*"))
    return matches[0] if matches else None


# PUBLIC_INTERFACE
@app.get(
    "/",
    tags=["Health"],
    summary="Health check",
    description="Basic health check endpoint for service monitoring.",
    operation_id="health_check",
)
def health_check():
    """Health check endpoint.

    Returns:
        JSON object with a basic status message.
    """
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.get(
    "/templates",
    response_model=List[TemplateItem],
    tags=["Templates"],
    summary="List meme templates",
    description="Returns the list of available meme templates and their metadata.",
    operation_id="list_templates",
)
def list_templates():
    """List templates available on the server."""
    items: List[TemplateItem] = []
    for fp in sorted(TEMPLATES_DIR.glob("*")):
        if fp.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            with Image.open(fp) as img:
                w, h = img.size
        except Exception:
            # If a template is corrupted, skip it rather than breaking listing.
            continue

        tid = fp.stem
        name = tid.replace("-", " ").replace("_", " ").title()
        items.append(
            TemplateItem(
                id=tid,
                name=name,
                image_url=f"/static/templates/{fp.name}",
                width=w,
                height=h,
            )
        )
    return items


# PUBLIC_INTERFACE
@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Uploads"],
    summary="Upload a custom image",
    description=(
        "Upload a custom image (PNG/JPG/WebP). Returns an upload_id and URL that can be used "
        "to generate memes from the uploaded image."
    ),
    operation_id="upload_image",
)
async def upload_image(file: UploadFile = File(...)):
    """Upload a custom image to be used for meme generation.

    Args:
        file: Multipart upload file.

    Returns:
        UploadResponse containing upload_id and image_url.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    upload_id = uuid.uuid4().hex
    safe_name = _safe_filename(file.filename or "upload.png")
    out_path = UPLOADS_DIR / f"{upload_id}_{safe_name}"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    # Validate image by attempting to open it.
    try:
        with Image.open(Path(out_path)) as _:
            pass
    except Exception:
        # Need to write first before open in some environments; validate via BytesIO instead.
        from io import BytesIO

        try:
            with Image.open(BytesIO(data)) as im:
                im.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    out_path.write_bytes(data)

    return UploadResponse(upload_id=upload_id, image_url=f"/static/uploads/{out_path.name}")


def _load_source_image(req: GenerateMemeRequest) -> Tuple[Image.Image, str]:
    """Load the source image for meme generation and return (PIL_image, source_label)."""
    if bool(req.template_id) == bool(req.upload_id):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of template_id or upload_id.",
        )

    if req.template_id:
        fp = _find_template_file(req.template_id)
        if not fp:
            raise HTTPException(status_code=404, detail="Template not found.")
        return Image.open(fp).convert("RGB"), f"template:{req.template_id}"

    fp = _find_upload_file(req.upload_id or "")
    if not fp:
        raise HTTPException(status_code=404, detail="Upload not found.")
    return Image.open(fp).convert("RGB"), f"upload:{req.upload_id}"


# PUBLIC_INTERFACE
@app.post(
    "/generate",
    response_model=GenerateMemeResponse,
    tags=["Memes"],
    summary="Generate a meme image",
    description=(
        "Generates a meme image by overlaying top and bottom text on either a selected template "
        "or an uploaded image. Returns a URL to the generated meme."
    ),
    operation_id="generate_meme",
)
def generate_meme(payload: GenerateMemeRequest):
    """Generate a meme from a template or uploaded image.

    Args:
        payload: GenerateMemeRequest containing either template_id or upload_id and text fields.

    Returns:
        GenerateMemeResponse with meme_id and image_url.
    """
    try:
        img, _label = _load_source_image(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to load source image: {e}") from e

    # Apply meme text.
    try:
        img = _draw_meme_text(img, payload.top_text or "", payload.bottom_text or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render text: {e}") from e

    meme_id = uuid.uuid4().hex
    out_name = f"{meme_id}.png"
    out_path = GENERATED_DIR / out_name

    try:
        img.save(out_path, format="PNG")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save meme: {e}") from e
    finally:
        try:
            img.close()
        except Exception:
            pass

    return GenerateMemeResponse(meme_id=meme_id, image_url=f"/static/generated/{out_name}")


# PUBLIC_INTERFACE
@app.get(
    "/download/{meme_id}",
    tags=["Memes"],
    summary="Download a generated meme image",
    description="Downloads a generated meme by meme_id as an attachment.",
    operation_id="download_meme",
)
def download_meme(meme_id: str):
    """Download a previously generated meme image.

    Args:
        meme_id: The meme identifier returned from /generate.

    Returns:
        PNG file response as an attachment.
    """
    fp = GENERATED_DIR / f"{meme_id}.png"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Meme not found.")
    return FileResponse(
        str(fp),
        media_type="image/png",
        filename=f"meme-{meme_id}.png",
    )


# PUBLIC_INTERFACE
@app.get(
    "/docs/web",
    tags=["Health"],
    summary="Frontend integration notes",
    description="Quick notes for how the frontend should call this API.",
    operation_id="web_integration_help",
)
def web_integration_help():
    """Provide basic integration notes for the web frontend."""
    return JSONResponse(
        {
            "base_url": "Use the same origin if proxied; otherwise set REACT_APP_API_BASE_URL in frontend.",
            "endpoints": {
                "GET /templates": "List templates",
                "POST /upload": "Upload an image (multipart form-data field 'file')",
                "POST /generate": "Generate meme from template_id or upload_id + top_text/bottom_text",
                "GET /download/{meme_id}": "Download generated meme as attachment",
                "/static/templates/...": "Template images",
                "/static/uploads/...": "Uploaded images",
                "/static/generated/...": "Generated memes",
            },
        }
    )
