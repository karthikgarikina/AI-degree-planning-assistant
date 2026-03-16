import pdfplumber
import re
from models import Course


def parse_transcript(path: str):

    courses = []

    pattern = r"[A-Z]{2,4}\s?\d{3}"

    with pdfplumber.open(path) as pdf:

        text = ""

        for page in pdf.pages:
            text += page.extract_text() + "\n"

    matches = re.findall(pattern, text)

    for code in matches:

        courses.append(
            Course(
                course_code=code.replace(" ", ""),
                title="Completed Course",
                credits=3,
                schedule=[]
            )
        )

    return courses