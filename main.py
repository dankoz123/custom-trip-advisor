from datetime import datetime, timedelta
from pathlib import Path
from crewai import Crew, Process

from core.config import MODEL
from core.agents import make_researcher, make_optimizer, make_planner
from core.tasks import make_research_task, make_optimization_task, make_planning_task
from core.display import show_itinerary, show_usage
from core.models import Itinerary


# ── User input ─────────────────────────────────────────────────────────────────

def prompt_destination() -> str:
    while True:
        val = input("Destination (e.g. Japan, Italy): ").strip()
        if val:
            return val
        print("  Destination cannot be empty.")


def prompt_date(label: str) -> datetime:
    while True:
        val = input(f"{label} (YYYY-MM-DD): ").strip()
        try:
            return datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            print("  Invalid date — use YYYY-MM-DD format.")


def prompt_hours_per_day() -> float:
    while True:
        val = input("Hours to explore per day [default: 8]: ").strip()
        if not val:
            return 8.0
        try:
            h = float(val)
            if 1 <= h <= 12:
                return h
            print("  Enter a number between 1 and 12.")
        except ValueError:
            print("  Please enter a number.")


def prompt_explore_days(total_days: int) -> int:
    while True:
        val = input(f"Days to explore [default: all {total_days}]: ").strip()
        if not val:
            return total_days
        try:
            n = int(val)
            if 1 <= n <= total_days:
                return n
            print(f"  Enter a number between 1 and {total_days}.")
        except ValueError:
            print("  Please enter a whole number.")


print("\n=== Custom Trip Advisor ===\n")

destination = prompt_destination()
start_dt    = prompt_date("Start date")
end_dt      = prompt_date("End date  ")

while end_dt < start_dt:
    print("  End date must be on or after start date.")
    end_dt = prompt_date("End date  ")

total_days    = (end_dt - start_dt).days + 1
explore       = prompt_explore_days(total_days)
hours_per_day = prompt_hours_per_day()

START_DATE = start_dt.strftime("%Y-%m-%d")
END_DATE   = end_dt.strftime("%Y-%m-%d")
date_range = [
    (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
    for i in range(explore)
]

print(f"\nPlanning {explore} day(s) in {destination}  ({START_DATE} → {END_DATE})\n")

# ── Agents ─────────────────────────────────────────────────────────────────────
researcher = make_researcher(destination)
optimizer  = make_optimizer(destination, explore, hours_per_day)
planner    = make_planner(destination, explore, START_DATE, hours_per_day)

# ── Tasks ──────────────────────────────────────────────────────────────────────
research_task     = make_research_task(researcher, destination, explore, date_range, hours_per_day)
optimization_task = make_optimization_task(optimizer, research_task, destination, explore, hours_per_day)
planning_task     = make_planning_task(
    planner, optimization_task, destination,
    START_DATE, END_DATE, explore, date_range, hours_per_day,
)

# ── Crew ───────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[researcher, optimizer, planner],
    tasks=[research_task, optimization_task, planning_task],
    process=Process.sequential,
    verbose=True,
)

# ── Run ────────────────────────────────────────────────────────────────────────
result = crew.kickoff()

# ── Display ────────────────────────────────────────────────────────────────────
itinerary: Itinerary = result.pydantic
show_itinerary(itinerary)
show_usage(result.token_usage, MODEL)

# ── Save ───────────────────────────────────────────────────────────────────────
output_file = f"itinerary_{destination.replace(' ', '_')}_{START_DATE}.json"
itinerary_dir = Path(__file__).parent / "itinerary"
itinerary_dir.mkdir(exist_ok=True)
(itinerary_dir / output_file).write_text(itinerary.model_dump_json(indent=2))
print(f"\nItinerary saved → itinerary/{output_file}")
