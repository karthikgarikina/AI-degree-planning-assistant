from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from loguru import logger

from config import settings
from models import CatalogCourse, ProgramRequirements


class MockCatalogClient:
    def __init__(self, catalog_path: str, requirements_path: str) -> None:
        self.catalog_path = Path(catalog_path)
        self.requirements_path = Path(requirements_path)

    @lru_cache(maxsize=1)
    def load_catalog(self) -> List[CatalogCourse]:
        with self.catalog_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [CatalogCourse(**course) for course in payload]

    @lru_cache(maxsize=1)
    def load_requirements(self) -> Dict[str, dict]:
        with self.requirements_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


client = MockCatalogClient(settings.catalog_path, settings.requirements_path)


def _log_tool_call(tool_name: str, tool_args: Dict[str, object]) -> None:
    logger.bind(tool_name=tool_name, tool_args=tool_args).info("Agent decided to call tool")


def search_courses(query: str, semester: str) -> List[dict]:
    """Search the mock university catalog for courses matching a query in a specific semester.

    Args:
        query: A course code, keyword, or title fragment.
        semester: The semester to search, such as "Fall 2026".

    Returns:
        A list of matching catalog entries as dictionaries.
    """

    _log_tool_call("search_courses", {"query": query, "semester": semester})
    normalized_query = query.lower().strip()
    results: List[dict] = []
    for course in client.load_catalog():
        haystacks = [course.course_code.lower(), course.title.lower(), " ".join(tag.lower() for tag in course.tags)]
        if semester not in course.semesters_offered:
            continue
        if any(normalized_query in haystack for haystack in haystacks):
            results.append(course.dict())
    return results


def get_course_details(course_code: str) -> dict:
    """Return a single course definition from the mock university catalog.

    Args:
        course_code: The exact course code to retrieve.

    Returns:
        A course dictionary if found, otherwise an empty dictionary.
    """

    _log_tool_call("get_course_details", {"course_code": course_code})
    for course in client.load_catalog():
        if course.course_code == course_code:
            return course.dict()
    return {}


def check_prerequisites(course_code: str, completed_courses: List[str]) -> dict:
    """Check whether a student has met the prerequisites for a course.

    Args:
        course_code: The course being evaluated.
        completed_courses: Completed course codes available to satisfy prerequisites.

    Returns:
        A dictionary containing whether the prerequisites are satisfied and which are missing.
    """

    _log_tool_call(
        "check_prerequisites",
        {"course_code": course_code, "completed_courses": completed_courses},
    )
    course = None
    for catalog_course in client.load_catalog():
        if catalog_course.course_code == course_code:
            course = catalog_course.dict()
            break

    if not course:
        return {"course_code": course_code, "satisfied": False, "missing_prerequisites": ["COURSE_NOT_FOUND"]}

    completed_lookup = set(completed_courses)
    missing = [item for item in course.get("prerequisites", []) if item not in completed_lookup]
    return {
        "course_code": course_code,
        "satisfied": not missing,
        "missing_prerequisites": missing,
    }


def get_program_requirements(program_name: str) -> dict:
    """Retrieve degree or minor requirements from the local mock MCP dataset.

    Args:
        program_name: The degree program or minor to evaluate.

    Returns:
        A dictionary describing required courses and elective rules.
    """

    _log_tool_call("get_program_requirements", {"program_name": program_name})
    requirements = client.load_requirements()
    payload = requirements.get(program_name)
    if not payload:
        return {}
    return ProgramRequirements(program_name=program_name, **payload).dict()
