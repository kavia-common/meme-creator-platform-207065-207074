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
THUMBNAILS_DIR = ASSET_ROOT / "thumbnails"
UPLOADS_DIR = ASSET_ROOT / "uploads"
GENERATED_DIR = ASSET_ROOT / "generated"

_ensure_dir(TEMPLATES_DIR)
_ensure_dir(THUMBNAILS_DIR)
_ensure_dir(UPLOADS_DIR)
_ensure_dir(GENERATED_DIR)


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


def _render_template_image(
    template_id: str, size: tuple[int, int], title: str, subtitle: str, theme: str
) -> Image.Image:
    """
    Render a simple bundled meme template image (retro, but distinct per theme).

    This generates deterministic visuals (no external binary assets needed) and allows us to
    also bundle/produce thumbnails on first run.
    """
    w, h = size
    img = Image.new("RGB", (w, h), (12, 10, 26))
    draw = ImageDraw.Draw(img)

    if theme == "drake":
        # Two-panel: top "NOPE" bottom "YEP"
        top_h = h // 2
        draw.rectangle([0, 0, w, top_h], fill=(40, 18, 64))
        draw.rectangle([0, top_h, w, h], fill=(10, 40, 64))
        font_big = _load_best_font(size=max(34, int(h * 0.10)))
        font_small = _load_best_font(size=max(18, int(h * 0.045)))

        def center_text(y0: int, y1: int, text: str, font, fill):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((w - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill)

        center_text(0, top_h, "NOPE", font_big, (255, 62, 165))
        center_text(top_h, h, "YEP", font_big, (0, 245, 255))
        center_text(0, top_h, title, font_small, (255, 255, 255))
        center_text(top_h, h, subtitle, font_small, (255, 255, 255))

        # Divider
        draw.line([(0, top_h), (w, top_h)], fill=(255, 255, 255), width=3)

    elif theme == "galaxy":
        # Starfield + neon ring
        import random

        rnd = random.Random(template_id)  # deterministic per id
        draw.rectangle([0, 0, w, h], fill=(8, 8, 20))
        for _ in range(700):
            x = rnd.randrange(0, w)
            y = rnd.randrange(0, h)
            c = rnd.randrange(160, 255)
            draw.point((x, y), fill=(c, c, c))

        # Rings
        cx, cy = int(w * 0.55), int(h * 0.55)
        for r, col in [
            (int(min(w, h) * 0.32), (255, 62, 165)),
            (int(min(w, h) * 0.28), (0, 245, 255)),
            (int(min(w, h) * 0.24), (255, 240, 0)),
        ]:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=10)

        font = _load_best_font(size=max(34, int(h * 0.085)))
        text = title.upper()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((w - tw) / 2, int(h * 0.12) - th / 2),
            text,
            font=font,
            fill=(255, 255, 255),
            stroke_width=4,
            stroke_fill=(0, 0, 0),
        )

    elif theme == "wave":
        # Neon synthwave gradient + grid + sun
        top = (255, 62, 165)
        mid = (0, 245, 255)
        bottom = (10, 10, 26)
        for y in range(h):
            t = y / max(h - 1, 1)
            # blend top->mid->bottom
            if t < 0.55:
                tt = t / 0.55
                r = int(top[0] * (1 - tt) + mid[0] * tt)
                g = int(top[1] * (1 - tt) + mid[1] * tt)
                b = int(top[2] * (1 - tt) + mid[2] * tt)
            else:
                tt = (t - 0.55) / 0.45
                r = int(mid[0] * (1 - tt) + bottom[0] * tt)
                g = int(mid[1] * (1 - tt) + bottom[1] * tt)
                b = int(mid[2] * (1 - tt) + bottom[2] * tt)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Sun
        sun_r = int(min(w, h) * 0.17)
        sun_cx, sun_cy = int(w * 0.28), int(h * 0.34)
        draw.ellipse(
            [sun_cx - sun_r, sun_cy - sun_r, sun_cx + sun_r, sun_cy + sun_r],
            fill=(255, 240, 0),
        )

        # Grid (bottom half)
        grid_y0 = int(h * 0.52)
        for x in range(0, w, 60):
            draw.line([(x, grid_y0), (x, h)], fill=(255, 255, 255), width=2)
        for y in range(grid_y0, h, 60):
            draw.line([(0, y), (w, y)], fill=(255, 255, 255), width=2)

        # Title
        font = _load_best_font(size=max(36, int(h * 0.09)))
        text = title.upper()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((w - tw) / 2, int(h * 0.10)),
            text,
            font=font,
            fill=(255, 255, 255),
            stroke_width=5,
            stroke_fill=(0, 0, 0),
        )

    else:
        # Fallback: simple gradient + label
        c1 = (255, 62, 165)
        c2 = (0, 245, 255)
        for y in range(h):
            t = y / max(h - 1, 1)
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        font = _load_best_font(size=max(34, int(h * 0.08)))
        text = title.upper()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((w - tw) / 2, (h - th) / 2),
            text,
            font=font,
            fill=(10, 10, 10),
            stroke_width=6,
            stroke_fill=(255, 255, 255),
        )

    return img


def _ensure_bundled_templates_and_thumbnails() -> None:
    """
    Ensure a bundled set of templates (and thumbnails) exist.

    We do NOT overwrite user-supplied files; we only create missing bundled assets by name.
    This keeps the app functional and ensures consistent gallery thumbnails.
    """
    # Curated "bundled" templates (generated locally; no binary assets required).
    bundled = [
        {
            "id": "drake-hot-take",
            "filename": "drake-hot-take.png",
            "size": (900, 900),
            "title": "Hot Take",
            "subtitle": "But Make It Retro",
            "theme": "drake",
        },
        {
            "id": "synthwave-sun",
            "filename": "synthwave-sun.png",
            "size": (900, 600),
            "title": "Synthwave",
            "subtitle": "Sunset Grid",
            "theme": "wave",
        },
        {
            "id": "galaxy-brain",
            "filename": "galaxy-brain.png",
            "size": (900, 600),
            "title": "Galaxy Brain",
            "subtitle": "Cosmic Thoughts",
            "theme": "galaxy",
        },
        {
            "id": "retro-gradient",
            "filename": "retro-gradient.png",
            "size": (900, 600),
            "title": "Retro",
            "subtitle": "Gradient",
            "theme": "gradient",
        },
    ]

    thumb_size = (360, 240)

    for spec in bundled:
        tpl_path = TEMPLATES_DIR / spec["filename"]
        thumb_path = THUMBNAILS_DIR / spec["filename"]

        if not tpl_path.exists():
            img = _render_template_image(
                template_id=spec["id"],
                size=spec["size"],
                title=spec["title"],
                subtitle=spec["subtitle"],
                theme=spec["theme"],
            )
            img.save(tpl_path, format="PNG")
            try:
                img.close()
            except Exception:
                pass

        # Create thumbnail (if missing) by resizing the template.
        if not thumb_path.exists():
            try:
                with Image.open(tpl_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", thumb_size, (10, 10, 26))
                    # Center the resized image
                    x = (thumb_size[0] - im.size[0]) // 2
                    y = (thumb_size[1] - im.size[1]) // 2
                    canvas.paste(im, (x, y))
                    canvas.save(thumb_path, format="PNG")
            except Exception:
                # If thumbnail generation fails, ignore and allow frontend to fall back to full image.
                continue


# Ensure bundled assets exist early so /templates always has content.
_ensure_bundled_templates_and_thumbnails()


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

# Static serving for templates/uploads/generated/thumbnails
app.mount("/static/templates", StaticFiles(directory=str(TEMPLATES_DIR)), name="templates")
app.mount(
    "/static/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails"
)
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount(
    "/static/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated"
)


class TemplateItem(BaseModel):
    """Template metadata returned to the frontend."""

    id: str = Field(..., description="Template identifier (filename without extension).")
    name: str = Field(..., description="Human-friendly template name.")
    image_url: str = Field(..., description="Absolute or relative URL to the template image.")
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="Optional URL to a smaller thumbnail image for gallery display.",
    )
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

        thumb = THUMBNAILS_DIR / fp.name
        thumbnail_url = f"/static/thumbnails/{thumb.name}" if thumb.exists() else None

        items.append(
            TemplateItem(
                id=tid,
                name=name,
                image_url=f"/static/templates/{fp.name}",
                thumbnail_url=thumbnail_url,
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
                "GET /templates": "List templates (includes thumbnail_url when available)",
                "POST /upload": "Upload an image (multipart form-data field 'file')",
                "POST /generate": "Generate meme from template_id or upload_id + top_text/bottom_text",
                "GET /download/{meme_id}": "Download generated meme as attachment",
                "/static/templates/...": "Template images",
                "/static/thumbnails/...": "Template thumbnails",
                "/static/uploads/...": "Uploaded images",
                "/static/generated/...": "Generated memes",
            },
        }
    )
