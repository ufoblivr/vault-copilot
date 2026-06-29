"""
Test suite for the OCR pipeline: preprocessing, confidence scoring,
perceptual hashing, JSON extraction, and failure handling.
"""
import json
from unittest.mock import patch

import pytest
from PIL import Image

from src.ocr.pipeline import OCRPipeline, ReceiptSchema, _extract_json


# ======================================================================
# ReceiptSchema validation
# ======================================================================

class TestReceiptSchema:
    def test_valid_full_data(self):
        schema = ReceiptSchema(
            store="Whole Foods", date="2024-01-15", total=45.99,
            category="groceries",
            items=[{"name": "milk"}, {"name": "bread"}],
        )
        assert schema.store == "Whole Foods"
        assert schema.total == 45.99
        assert len(schema.items) == 2

    def test_defaults_applied(self):
        schema = ReceiptSchema()
        assert schema.store == "Unknown"
        assert schema.total == 0.0
        assert schema.items == []
        assert schema.extraction_failed is False
        assert schema.ocr_confidence == 0.0

    def test_metadata_fields_present(self):
        schema = ReceiptSchema(
            ocr_confidence=0.85, low_confidence=False,
            phash="abcdef", raw_ocr_text="some text",
        )
        assert schema.ocr_confidence == 0.85
        assert schema.phash == "abcdef"
        assert schema.raw_ocr_text == "some text"

    def test_extraction_failed_flag(self):
        schema = ReceiptSchema(extraction_failed=True)
        assert schema.extraction_failed is True


# ======================================================================
# JSON extraction helper
# ======================================================================

class TestExtractJSON:
    def test_clean_json(self):
        result = _extract_json('{"store": "Target", "total": 25.0}')
        assert result["store"] == "Target"

    def test_json_with_surrounding_text(self):
        text = 'Here is the data: {"store": "Target", "total": 25.0} done.'
        result = _extract_json(text)
        assert result is not None
        assert result["store"] == "Target"

    def test_json_in_code_fence(self):
        text = '```json\n{"store": "Walmart", "total": 12.0}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["store"] == "Walmart"

    def test_invalid_json_returns_none(self):
        result = _extract_json("this is not json at all")
        assert result is None

    def test_empty_string(self):
        result = _extract_json("")
        assert result is None

    def test_nested_json(self):
        text = '{"store": "S", "items": [{"name": "milk"}], "total": 5.0}'
        result = _extract_json(text)
        assert result is not None
        assert result["items"] == [{"name": "milk"}]


# ======================================================================
# Image preprocessing
# ======================================================================

class TestPreprocessImage:
    def test_preprocess_creates_file(self, test_image):
        from pathlib import Path
        result_path = OCRPipeline._preprocess_image(test_image)
        assert Path(result_path).exists()
        # Verify it's a valid image
        img = Image.open(result_path)
        assert img.mode == "L"  # Grayscale
        img.close()  # Release file handle before unlink (Windows)
        # Cleanup
        Path(result_path).unlink(missing_ok=True)

    def test_preprocess_handles_rgba(self, tmp_path):
        from pathlib import Path
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        path = str(tmp_path / "rgba_test.png")
        img.save(path)
        result_path = OCRPipeline._preprocess_image(path)
        assert Path(result_path).exists()
        Path(result_path).unlink(missing_ok=True)


# ======================================================================
# Perceptual hash
# ======================================================================

class TestPerceptualHash:
    def test_hash_returns_string(self, test_image):
        phash = OCRPipeline._compute_phash(test_image)
        assert isinstance(phash, str)
        assert len(phash) > 0

    def test_same_image_same_hash(self, test_image):
        h1 = OCRPipeline._compute_phash(test_image)
        h2 = OCRPipeline._compute_phash(test_image)
        assert h1 == h2

    def test_different_images_different_hash(self, test_image, tmp_path):
        img = Image.new("RGB", (200, 200), color="blue")
        other_path = str(tmp_path / "other.png")
        img.save(other_path)
        h1 = OCRPipeline._compute_phash(test_image)
        h2 = OCRPipeline._compute_phash(other_path)
        assert h1 != h2


# ======================================================================
# Full pipeline (with mock LLM)
# ======================================================================

class TestProcessImage:
    def test_successful_extraction(self, test_image, mock_llm_pipeline):
        """Pipeline should return a dict with store, total, confidence, phash."""
        pipeline = OCRPipeline()
        result = pipeline.process_image(test_image, mock_llm_pipeline)

        assert isinstance(result, dict)
        assert "store" in result
        assert "total" in result
        assert "ocr_confidence" in result
        assert "phash" in result
        assert "extraction_failed" in result
        assert isinstance(result["ocr_confidence"], float)
        assert isinstance(result["phash"], str)

    def test_extraction_failure_returns_flag(self, test_image):
        """When LLM returns garbage, extraction_failed should be True."""
        def bad_pipeline(prompt, **kwargs):
            return [{"generated_text": "GARBAGE NOT JSON"}]

        pipeline = OCRPipeline()
        result = pipeline.process_image(test_image, bad_pipeline)
        assert result["extraction_failed"] is True
        assert "raw_ocr_text" in result

    def test_phash_always_present(self, test_image, mock_llm_pipeline):
        pipeline = OCRPipeline()
        result = pipeline.process_image(test_image, mock_llm_pipeline)
        assert result["phash"] != ""

    def test_confidence_always_present(self, test_image, mock_llm_pipeline):
        pipeline = OCRPipeline()
        result = pipeline.process_image(test_image, mock_llm_pipeline)
        assert "ocr_confidence" in result
        assert "low_confidence" in result
