from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from agent import build_graph, build_initial_state
from config import settings
from models import DegreePlan
from pdf_parser import parse_transcript


def configure_logging() -> Path:
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "agent.log"

    logger.remove()
    if log_path.exists():
        log_path.write_text("", encoding="utf-8")

    def json_sink(message) -> None:
        record = message.record
        payload: Dict[str, Any] = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "event": record["message"],
        }
        payload.update(record["extra"])
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")

    logger.add(json_sink, level=settings.log_level)
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    )
    return log_path


def save_degree_plan(degree_plan: DegreePlan) -> Path:
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "degree_plan.json"
    with output_path.open("w", encoding="utf-8") as handle:
        payload = degree_plan.model_dump() if hasattr(degree_plan, "model_dump") else degree_plan.dict()
        json.dump(payload, handle, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous university degree planner")
    parser.add_argument("--transcript", required=True, help="Path to a student transcript PDF.")
    parser.add_argument("--goal", required=True, help="Degree planning goal for the agent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    print("[1/4] Preparing degree planner...")
    logger.bind(goal=args.goal, transcript=args.transcript).info("Starting degree planning run")

    print(f"[2/4] Reading transcript: {args.transcript}")
    transcript = parse_transcript(args.transcript)
    logger.bind(
        student_id=transcript.student_id,
        completed_course_codes=[course.course_code for course in transcript.completed_courses],
        completed_course_count=len(transcript.completed_courses),
    ).info("Transcript parsed successfully")

    degree_plan = DegreePlan(
        student_id=transcript.student_id,
        completed_courses=transcript.completed_courses,
        planned_semesters=[],
    )

    print("[3/4] Generating degree plan...")
    state = build_initial_state(goal=args.goal, degree_plan=degree_plan)
    graph = build_graph()
    result = graph.invoke(state)

    saved_path = save_degree_plan(result["degree_plan"])
    logger.bind(
        output_path=str(saved_path),
        planned_semester_count=len(result["degree_plan"].planned_semesters),
        planning_notes=result["planning_notes"],
        final_response=result["final_response"],
    ).info("Degree plan saved")
    print(f"[4/4] Plan generated. Available at: {saved_path}")


if __name__ == "__main__":
    main()
