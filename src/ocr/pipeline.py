"""
OCR pipeline for receipt image processing.

Handles image preprocessing, text extraction via EasyOCR, perceptual
hash computation for deduplication, and LLM-based structured data
extraction with confidence scoring and robust error handling.
"""

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import easyocr
import imagehash
import torch
from loguru import logger
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field, ValidationError

from src.config import LLM_MAX_NEW_TOKENS, OCR_CONFIDENCE_THRESHOLD, OCR_MAX_TEXT_LENGTH


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ReceiptItem(BaseModel):
    """A single line-item extracted from a receipt."""

    name: str


class ReceiptSchema(BaseModel):
    """Validated structure for receipt extraction results.

    Fields beyond the core receipt data carry OCR metadata so that
    callers can make informed accept/reject decisions.
    """

    store: str = Field(default="Unknown")
    date: str = Field(default="Unknown")
    total: float = Field(default=0.0)
    category: str = Field(default="Unknown")
    items: list[ReceiptItem] = Field(default_factory=list)
    ocr_confidence: float = Field(default=0.0)
    low_confidence: bool = Field(default=False)
    extraction_failed: bool = Field(default=False)
    phash: str = Field(default="")
    raw_ocr_text: str = Field(default="")


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """Attempt to fix common JSON syntax issues from LLM output."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single quotes with double quotes (but not apostrophes in words)
    text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '"', text)
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    # Fix unquoted keys: {store: "x"} -> {"store": "x"}
    text = re.sub(r'{\s*(\w+)\s*:', r'{"\1":', text)
    text = re.sub(r',\s*(\w+)\s*:', r', "\1":', text)
    return text


def _extract_fields_regex(text: str) -> dict[str, Any] | None:
    """Last-resort regex extraction of individual receipt fields from LLM output."""
    result: dict[str, Any] = {}

    # Store
    m = re.search(r'"?store"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        result["store"] = m.group(1)

    # Date
    m = re.search(r'"?date"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        result["date"] = m.group(1)

    # Total — look for number after "total"
    m = re.search(r'"?total"?\s*[:=]\s*"?(\d+\.?\d*)"?', text, re.IGNORECASE)
    if m:
        try:
            result["total"] = float(m.group(1))
        except ValueError:
            pass

    # Category
    m = re.search(r'"?category"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        result["category"] = m.group(1)

    # Items — try to find a list
    m = re.search(r'"?items"?\s*[:=]\s*\[([^\]]*)\]', text, re.IGNORECASE)
    if m:
        items_str = m.group(1)
        items = re.findall(r'"([^"]+)"', items_str)
        if items:
            result["items"] = items

    # Only return if we got at least store or total
    if "store" in result or "total" in result:
        return result
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    """Attempt to extract a JSON object from *text* using multiple strategies.

    Strategies (tried in order):
        1. Direct ``json.loads`` on the full text.
        2. Locate the outermost ``{ … }`` pair and parse that substring.
        3. Repair common JSON errors and retry.
        4. Regex search for a fenced JSON code-block.
        5. Regex extraction of individual fields (last resort).

    Returns:
        Parsed ``dict`` on success, ``None`` if every strategy fails.
    """

    # Strategy 1 – direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2 – outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = text[start : end + 1]
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 3 – repair and retry
        repaired = _repair_json(json_candidate)
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4 – fenced code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 5 – regex field extraction (last resort)
    data = _extract_fields_regex(text)
    if data is not None:
        logger.info("Used regex fallback to extract {} fields", len(data))
        return data

    return None


# ---------------------------------------------------------------------------
# OCR Pipeline
# ---------------------------------------------------------------------------

class OCRPipeline:
    """End-to-end pipeline: preprocess → OCR → LLM extraction → validation.

    Attributes:
        reader: The EasyOCR ``Reader`` instance (loaded once at init).
    """

    def __init__(self) -> None:
        use_gpu = torch.cuda.is_available()
        logger.info("Loading EasyOCR model (GPU={})", use_gpu)
        self.reader = easyocr.Reader(
            ["en"], gpu=use_gpu, verbose=False
        )
        logger.info("EasyOCR model loaded successfully")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_image(image_path: str) -> str:
        """Apply a preprocessing pipeline to improve OCR accuracy.

        Steps:
            1. Auto-orient using EXIF data.
            2. Convert to grayscale.
            3. Auto-contrast with a 2 % cutoff.
            4. Sharpen.

        Args:
            image_path: Filesystem path to the source image.

        Returns:
            Path to a temporary preprocessed image file (PNG).
        """
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)
        img = img.filter(ImageFilter.SHARPEN)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        logger.debug(
            "Preprocessed image saved to {}", tmp.name
        )
        return tmp.name

    # ------------------------------------------------------------------
    # Perceptual hash
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phash(image_path: str) -> str:
        """Compute a perceptual hash of the *original* image.

        Args:
            image_path: Filesystem path to the source image.

        Returns:
            Hex-string perceptual hash.
        """
        img = Image.open(image_path)
        return str(imagehash.phash(img))

    # ------------------------------------------------------------------
    # LLM extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(raw_text: str) -> str:
        """Build the ChatML-formatted extraction prompt."""
        return (
            "<|im_start|>system\n"
            "Extract strictly as JSON. "
            "Keys: store, date, total, category, items. "
            "Items is a list of strings. "
            "Respond ONLY with valid JSON, no extra text."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{raw_text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _build_retry_prompt(raw_text: str) -> str:
        """Build a simplified prompt for the retry attempt."""
        return (
            "<|im_start|>system\n"
            "Return a JSON object with keys: store, date, total, category, items. "
            "total must be a number. items is a list of item name strings. "
            "Output ONLY the JSON object."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{raw_text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _call_llm(self, prompt: str, llm_pipeline: Any) -> str:
        """Invoke the LLM pipeline and return the generated text.

        The opening brace from the prompt is prepended so the LLM
        output forms a complete JSON object.
        """
        result = llm_pipeline(
            prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            return_full_text=False,
        )
        return "{" + result[0]["generated_text"]

    # ------------------------------------------------------------------
    # Core public method
    # ------------------------------------------------------------------

    def process_image(self, image_path: str, llm_pipeline: Any) -> dict[str, Any]:
        """Process a receipt image and return structured data.

        Pipeline:
            1. Compute perceptual hash on the original image.
            2. Preprocess (orient, grayscale, contrast, sharpen).
            3. Run EasyOCR with confidence scoring.
            4. Call LLM for JSON extraction (with retry on failure).
            5. Validate via Pydantic and return enriched dict.

        On extraction failure the returned dict contains
        ``extraction_failed=True`` together with the raw OCR text,
        confidence, and hash so the caller can decide what to do.

        Args:
            image_path: Path to the receipt image file.
            llm_pipeline: A HuggingFace-style text-generation pipeline.

        Returns:
            Dictionary matching ``ReceiptSchema`` with OCR metadata.
        """
        t_start = time.perf_counter()
        logger.info("Processing image: {}", Path(image_path).name)

        # 1. Perceptual hash (on original, before preprocessing)
        phash = self._compute_phash(image_path)
        logger.debug("Perceptual hash: {}", phash)

        # 2. Preprocess
        preprocessed_path = self._preprocess_image(image_path)

        # 3. OCR with confidence
        detections: list[tuple[list, str, float]] = self.reader.readtext(
            preprocessed_path, detail=1
        )

        if detections:
            texts = [det[1] for det in detections]
            confidences = [det[2] for det in detections]
            raw_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            raw_text = ""
            avg_confidence = 0.0

        low_confidence = avg_confidence < OCR_CONFIDENCE_THRESHOLD

        logger.info(
            "OCR complete: {} text boxes, avg_confidence={:.3f}, low_confidence={}",
            len(detections),
            avg_confidence,
            low_confidence,
        )

        # Clean up temp file
        try:
            Path(preprocessed_path).unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove temp preprocessed image")

        if not raw_text.strip():
            elapsed = time.perf_counter() - t_start
            logger.warning(
                "No text detected in image (took {:.2f}s)", elapsed
            )
            return ReceiptSchema(
                extraction_failed=True,
                raw_ocr_text="",
                ocr_confidence=avg_confidence,
                low_confidence=low_confidence,
                phash=phash,
            ).model_dump()

        # 4. LLM extraction with retry
        # Truncate OCR text to avoid slow CPU inference on dense receipts
        llm_text = raw_text
        if len(llm_text) > OCR_MAX_TEXT_LENGTH:
            logger.info(
                "Truncating OCR text from {} to {} chars for LLM",
                len(llm_text), OCR_MAX_TEXT_LENGTH,
            )
            llm_text = llm_text[:OCR_MAX_TEXT_LENGTH]

        data: dict[str, Any] | None = None
        for attempt, prompt_fn in enumerate(
            [self._build_prompt, self._build_retry_prompt], start=1
        ):
            prompt = prompt_fn(llm_text)
            try:
                llm_output = self._call_llm(prompt, llm_pipeline)
                data = _extract_json(llm_output)
                if data is not None:
                    logger.debug(
                        "JSON extraction succeeded on attempt {}", attempt
                    )
                    break
                logger.warning(
                    "JSON extraction returned None on attempt {}", attempt
                )
                logger.debug(
                    "LLM raw output (first 500 chars): {}",
                    llm_output[:500] if llm_output else "<empty>",
                )
                # If first attempt returned text but JSON parse failed,
                # skip retry — it won't help and doubles CPU time
                if attempt == 1 and llm_output and len(llm_output) > 5:
                    logger.info("Skipping retry — LLM produced output but JSON was invalid")
                    break
            except Exception as exc:
                logger.warning(
                    "LLM call failed on attempt {}: {}", attempt, exc
                )

        if data is None:
            elapsed = time.perf_counter() - t_start
            logger.error(
                "All extraction attempts failed for {} (took {:.2f}s)",
                Path(image_path).name,
                elapsed,
            )
            return ReceiptSchema(
                extraction_failed=True,
                raw_ocr_text=raw_text,
                ocr_confidence=avg_confidence,
                low_confidence=low_confidence,
                phash=phash,
            ).model_dump()

        # 5. Normalise items list (LLM may return plain strings)
        items_raw = data.get("items")
        if items_raw and isinstance(items_raw, list):
            if items_raw and isinstance(items_raw[0], str):
                data["items"] = [{"name": i} for i in items_raw]

        # Inject OCR metadata
        data["ocr_confidence"] = avg_confidence
        data["low_confidence"] = low_confidence
        data["phash"] = phash
        data["raw_ocr_text"] = raw_text
        data["extraction_failed"] = False

        # 6. Validate with Pydantic
        try:
            validated = ReceiptSchema(**data)
        except ValidationError as exc:
            elapsed = time.perf_counter() - t_start
            logger.error(
                "Pydantic validation failed (took {:.2f}s): {}", elapsed, exc
            )
            return ReceiptSchema(
                extraction_failed=True,
                raw_ocr_text=raw_text,
                ocr_confidence=avg_confidence,
                low_confidence=low_confidence,
                phash=phash,
            ).model_dump()

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Extraction succeeded for {} — store={!r}, total={}, items={}, "
            "confidence={:.3f} (took {:.2f}s)",
            Path(image_path).name,
            validated.store,
            validated.total,
            len(validated.items),
            avg_confidence,
            elapsed,
        )

        return validated.model_dump()