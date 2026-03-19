from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pdfplumber

from models import Course, ParsedTranscript

STUDENT_ID_PATTERN = re.compile(r"Student ID:\s*([A-Z0-9-]+)")
PIPE_DELIMITED_PATTERN = re.compile(
    r"^(?P<term>[A-Za-z]+\s+\d{4})\s*\|\s*(?P<code>[A-Z]{2,4}\s?\d{3,4})\s*\|\s*"
    r"(?P<title>.+?)\s*\|\s*(?P<grade>[A-F][+-]?|P)\s*\|\s*(?P<credits>\d+)$"
)
FALLBACK_PATTERN = re.compile(
    r"(?P<code>[A-Z]{2,4}\s?\d{3,4})\s+(?P<title>.+?)\s+(?P<grade>[A-F][+-]?|P)\s+(?P<credits>\d+)$"
)


def _extract_text_from_pdf(path: Path) -> str:
    text_blocks: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_blocks.append(page_text)
    return "\n".join(text_blocks)


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _extract_text_from_pdf(path)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported transcript format: {path.suffix}")


def parse_transcript(path: str) -> ParsedTranscript:
    transcript_path = Path(path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    raw_text = _extract_text(transcript_path)
    student_id_match = STUDENT_ID_PATTERN.search(raw_text)
    student_id = student_id_match.group(1) if student_id_match else "UNKNOWN-STUDENT"

    completed_courses: List[Course] = []
    seen_course_codes = set()

    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        match = PIPE_DELIMITED_PATTERN.match(line) or FALLBACK_PATTERN.match(line)
        if not match:
            continue

        course_code = match.group("code").replace(" ", "")
        if course_code in seen_course_codes:
            continue

        seen_course_codes.add(course_code)
        completed_courses.append(
            Course(
                course_code=course_code,
                title=match.group("title").strip(),
                credits=int(match.group("credits")),
                schedule=[],
            )
        )

    if not completed_courses:
        raise ValueError("No completed courses were detected in the transcript.")

    return ParsedTranscript(student_id=student_id, completed_courses=completed_courses)
