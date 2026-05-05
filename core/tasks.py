from crewai import Task
from core.models import Itinerary


def make_research_task(agent, destination: str, explore_days: int,
                       date_range: list[str], hours_per_day: float) -> Task:
    max_min = int(hours_per_day * 60)
    return Task(
        description=(
            f"Research the best places to visit in {destination} for a {explore_days}-day trip "
            f"covering: {', '.join(date_range)}. "
            "For each attraction provide: name, city/region, approximate latitude and longitude, "
            "visit duration in minutes, and a real source URL. "
            f"Each day has a budget of {max_min} minutes of visiting time, so research enough "
            "attractions to comfortably fill that budget every day (aim for at least 20% more "
            "than needed so the optimizer has options to choose from)."
        ),
        expected_output=(
            "A list of attractions each with: name, city/region, lat/lon, duration (minutes), source URL."
        ),
        agent=agent,
    )


def make_optimization_task(agent, research_task: Task, destination: str,
                           explore_days: int, hours_per_day: float) -> Task:
    max_min = int(hours_per_day * 60)
    min_min = int(max_min * 0.9)
    return Task(
        description=(
            f"Using the research findings, group all attractions into {explore_days} days "
            f"for {destination} with zero backtracking:\n\n"
            "1. Cluster attractions by geographic proximity — nearby attractions go on the same day.\n"
            "2. Order the days so the journey flows logically in one direction "
            "(e.g. north to south, coast to interior) without revisiting an area.\n"
            "3. Within each day, sort attractions by walking/travel distance so the "
            "tourist moves in a single direction rather than zigzagging.\n"
            "4. If the trip spans multiple cities, dedicate full days to each city before moving on.\n"
            f"5. Each day's total visit duration must be between {min_min} and {max_min} minutes "
            f"({hours_per_day * 0.9:.1f}h – {hours_per_day:.1f}h). "
            "Add or remove attractions per day until the sum falls within this range.\n\n"
            "For each day, show the attractions in order with their durations and a running total, "
            "plus a one-line reason for the grouping."
        ),
        expected_output=(
            f"A {explore_days}-day optimized route plan. Each day lists ordered attractions "
            f"with durations summing to {min_min}–{max_min} minutes, plus grouping rationale."
        ),
        agent=agent,
        context=[research_task],
    )


def make_planning_task(agent, optimization_task: Task, destination: str,
                       start_date: str, end_date: str,
                       explore_days: int, date_range: list[str],
                       hours_per_day: float) -> Task:
    max_min = int(hours_per_day * 60)
    min_min = int(max_min * 0.9)
    schema  = Itinerary.model_json_schema()
    return Task(
        description=(
            f"Using the optimized route plan, build a {explore_days}-day itinerary for {destination} "
            f"covering: {', '.join(date_range)}.\n\n"
            "Return ONLY a JSON object — no prose, no markdown — matching this schema:\n"
            f"{schema}\n\n"
            "Rules:\n"
            f"- destination: \"{destination}\"\n"
            f"- start_date: \"{start_date}\", end_date: \"{end_date}\", total_days: {explore_days}\n"
            "- days: one entry per date in the date list, strictly following the optimizer's order\n"
            "- each day: day_number (1-based), date (YYYY-MM-DD), title (short, vivid), "
            "description (one sentence), attractions list\n"
            "- each attraction: time (HH:MM, start at 09:00 unless travel day), "
            "place (full name + city), duration_min (integer), source (full URL), "
            "lat (decimal latitude), lon (decimal longitude)\n"
            "- do NOT reorder attractions — respect the Route Optimizer's sequence exactly\n"
            "- schedule realistic travel time gaps between attractions in different areas\n"
            f"- HARD CONSTRAINT: each day's sum of duration_min must be between "
            f"{min_min} and {max_min} minutes — no exceptions"
        ),
        expected_output=(
            f"A single valid JSON object matching the Itinerary schema for {explore_days} days, "
            f"each day totalling {min_min}–{max_min} minutes of visits."
        ),
        output_pydantic=Itinerary,
        agent=agent,
        context=[optimization_task],
    )
