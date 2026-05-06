from crewai import Agent


def make_researcher(destination: str, llm) -> Agent:
    return Agent(
        role="Travel Researcher",
        goal=(
            f"Research the top attractions for {destination}. "
            "For every attraction provide: name, city/region, geographic coordinates "
            "(approximate latitude/longitude), recommended visit duration in minutes, "
            "and a real working source URL."
        ),
        backstory=(
            "You are an experienced travel journalist who has personally visited every "
            "corner of the world. You always back recommendations with real sources."
        ),
        llm=llm,
        verbose=True,
    )


def make_optimizer(destination: str, explore_days: int, hours_per_day: float, llm) -> Agent:
    max_min = int(hours_per_day * 60)
    min_min = int(max_min * 0.9)
    return Agent(
        role="Route Optimizer",
        goal=(
            f"Take the full list of attractions for {destination} and produce an optimized "
            f"{explore_days}-day grouping that eliminates backtracking. "
            "Cluster attractions by geographic proximity so that each day covers one "
            "contiguous area. Order the days so the overall journey flows in one direction "
            "without returning to places already visited. Within each day, order attractions "
            "by walking/travel distance to minimize time spent moving. "
            f"Each day's total visit time must be between {min_min} and {max_min} minutes — "
            "add or drop attractions per day to hit this range."
        ),
        backstory=(
            "You are a logistics expert and seasoned traveller who specialises in building "
            "efficient sightseeing routes. You think spatially — always asking 'what is "
            "closest to where we already are?' before moving on."
        ),
        llm=llm,
        verbose=True,
    )


def make_planner(destination: str, explore_days: int, start_date: str, hours_per_day: float, llm) -> Agent:
    max_min = int(hours_per_day * 60)
    min_min = int(max_min * 0.9)
    return Agent(
        role="Itinerary Planner",
        goal=(
            f"Using the optimized route, produce a {explore_days}-day itinerary "
            f"for {destination} starting {start_date}. "
            "Respect the day groupings and attraction order from the Route Optimizer exactly. "
            f"Each day's scheduled visits must total between {min_min} and {max_min} minutes. "
            "Output must be valid JSON matching the Itinerary schema — no extra text, no markdown."
        ),
        backstory=(
            "You are a meticulous trip-planning specialist who creates perfectly timed "
            "itineraries. You trust the Route Optimizer's ordering and focus on realistic "
            "timings, opening hours, and clean JSON output."
        ),
        llm=llm,
        verbose=True,
    )
