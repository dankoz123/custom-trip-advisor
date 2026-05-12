import math
import requests
from core.models import Itinerary


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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


def fetch_intercity_routes(itinerary: Itinerary, same_city_km: float = 30.0) -> dict:
    """Return {from_day_number: route} for consecutive days that are in different cities.

    Keyed by the day number you are travelling *from* (e.g. key 1 = end of Day 1 → start of Day 2).
    Days within same_city_km of each other are considered the same city and are skipped.
    """
    intercity = {}
    days = itinerary.days
    for i in range(len(days) - 1):
        day_a = days[i]
        day_b = days[i + 1]
        last_a  = next((a for a in reversed(day_a.attractions) if a.lat and a.lon), None)
        first_b = next((a for a in day_b.attractions         if a.lat and a.lon), None)
        if not last_a or not first_b:
            continue
        if _haversine_km(last_a.lat, last_a.lon, first_b.lat, first_b.lon) < same_city_km:
            continue
        route = fetch_day_route([last_a, first_b])
        if route:
            intercity[day_a.day_number] = {
                **route,
                "from_place": last_a.place,
                "to_place":   first_b.place,
            }
    return intercity
