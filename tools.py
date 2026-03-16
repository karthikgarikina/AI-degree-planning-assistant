import json
from loguru import logger


CATALOG_FILE = "data/course_catalog.json"


def load_catalog():

    with open(CATALOG_FILE) as f:
        return json.load(f)


def search_courses(query: str, semester: str):

    logger.info(
        "Agent decided to call tool",
        event="tool_call",
        tool_name="search_courses",
        tool_args={"query": query, "semester": semester},
    )

    catalog = load_catalog()

    results = []

    for c in catalog:

        if semester not in c["semester"]:
            continue

        if query.lower() in c["title"].lower() or query.lower() in c.get("tags", []):

            results.append(c)

    return results


def get_course(course_code: str):

    catalog = load_catalog()

    for c in catalog:
        if c["course_code"] == course_code:
            return c

    return None