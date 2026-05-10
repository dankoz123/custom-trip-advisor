import requests
from core.models import Itinerary


def fetch_day_route(attractions) -> dict | None:
    located = [a for a in attractions if a.lat is not None and a.lon is not None]
    if len(located) < 2:
        return None
    coords = ";".join(f"{a.lon},{a.lat}" for a in located)
    try:
        resp = requests.get(
            f"http://router.project-osrm.org/route/v1/driving/{coords}",
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") != "Ok":
            return None
        route = data["routes"][0]
        return {
            "geometry": route["geometry"]["coordinates"],
            "legs": [
                {"distance_m": leg["distance"], "duration_s": leg["duration"]}
                for leg in route["legs"]
            ],
        }
    except Exception:
        return None


def fetch_all_routes(itinerary: Itinerary) -> dict:
    routes = {}
    for day in itinerary.days:
        route = fetch_day_route(day.attractions)
        if route:
            routes[day.day_number] = route
    return routes
