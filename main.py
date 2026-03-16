import argparse
import json
from loguru import logger

from pdf_parser import parse_transcript
from models import DegreePlan
from agent import build_graph
from config import settings


logger.add(
    "output/agent.log",
    serialize=True,
    level=settings.LOG_LEVEL,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--transcript", required=True)
    parser.add_argument("--goal", required=True)

    args = parser.parse_args()

    completed = parse_transcript(args.transcript)

    degree_plan = DegreePlan(
        student_id="12345",
        completed_courses=completed,
    )

    graph = build_graph()

    state = {
        "goal": args.goal,
        "plan": degree_plan,
        "step": 0,
    }

    result = graph.invoke(state)

    with open("output/degree_plan.json", "w") as f:
        json.dump(result["plan"].dict(), f, indent=2)

    logger.info("Degree plan saved", event="plan_generated")


if __name__ == "__main__":
    main()