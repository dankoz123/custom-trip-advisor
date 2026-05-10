from pathlib import Path
import folium
from core.models import Itinerary

DAY_COLOURS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#E67E22",
    "#1ABC9C", "#E91E63", "#34495E", "#F39C12", "#00BCD4",
]


def day_label(day_number: int) -> str:
    if day_number <= 9:
        return str(day_number)
    return chr(ord("A") + day_number - 10)


def build_map(itinerary: Itinerary, routes: dict) -> folium.Map:
    all_lats = [a.lat for d in itinerary.days for a in d.attractions if a.lat is not None]
    all_lons = [a.lon for d in itinerary.days for a in d.attractions if a.lon is not None]
    centre = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)) if all_lats else (20, 0)

    m = folium.Map(location=centre, zoom_start=7, control_scale=True)

    for day in itinerary.days:
        colour = DAY_COLOURS[(day.day_number - 1) % len(DAY_COLOURS)]
        label  = day_label(day.day_number)

        if day.day_number in routes:
            road_coords = [(lat, lon) for lon, lat in routes[day.day_number]["geometry"]]
            total_drive_s = sum(l["duration_s"] for l in routes[day.day_number]["legs"])
            total_drive_min = round(total_drive_s / 60)
            folium.PolyLine(
                road_coords, color=colour, weight=4, opacity=0.8,
                tooltip=f"Day {day.day_number}: {day.title} · {total_drive_min} min driving",
            ).add_to(m)
        else:
            coords = [(a.lat, a.lon) for a in day.attractions if a.lat is not None and a.lon is not None]
            if len(coords) > 1:
                folium.PolyLine(coords, color=colour, weight=3, opacity=0.6,
                                tooltip=f"Day {day.day_number}: {day.title}").add_to(m)

        for attr in day.attractions:
            if attr.lat is None or attr.lon is None:
                continue
            folium.Marker(
                location=(attr.lat, attr.lon),
                popup=folium.Popup(
                    f"<b>{attr.place}</b><br>Day {day.day_number} · {attr.time}<br>{attr.duration_min} min",
                    max_width=220,
                ),
                tooltip=attr.place,
                icon=folium.DivIcon(
                    html=(
                        f'<div style="background:{colour};color:white;border-radius:50%;'
                        f'width:28px;height:28px;display:flex;align-items:center;'
                        f'justify-content:center;font-weight:bold;font-size:13px;'
                        f'border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,.4);">'
                        f'{label}</div>'
                    ),
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                ),
            ).add_to(m)

    return m


def save_map(itinerary: Itinerary, routes: dict, static_dir: Path) -> None:
    m = build_map(itinerary, routes)
    static_dir.mkdir(exist_ok=True)
    m.save(str(static_dir / "map.html"))
