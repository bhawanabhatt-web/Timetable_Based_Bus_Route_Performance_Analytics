
import argparse
import csv
import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from lxml import etree

API_KEY = os.environ.get("BODS_API_KEY")
if not API_KEY:
    raise SystemExit("Set the BODS_API_KEY environment variable first.")

DATAFEED_URL = "https://data.bus-data.dft.gov.uk/api/v1/datafeed/"
BOUNDING_BOX = "-1.75,51.55,-0.95,52.15"

OUT_PATH = Path("data/Raw Data/Location/location_pings.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "polled_at_utc",
    "recorded_at_time",
    "valid_until_time",
    "item_identifier",
    "vehicle_ref",
    "line_ref",
    "published_line_name",
    "operator_ref",
    "direction_ref",
    "data_frame_ref",
    "dated_vehicle_journey_ref",
    "origin_ref",
    "origin_name",
    "destination_ref",
    "destination_name",
    "origin_aimed_departure_time",
    "longitude",
    "latitude",
    "bearing",
    "block_ref",
    "vehicle_journey_ref",
]


def detect_namespace(root) -> dict:
    """The real BODS file may or may not declare a default namespace at the
    root (e.g. <Siri xmlns="...">). Detect it from the actual file instead
    of hardcoding a guessed URI, same approach as the timetable parser."""
    tag = root.tag
    if tag.startswith("{"):
        uri = tag[1:].split("}")[0]
        return {"siri": uri}
    return {}  # no namespace in use -- plain tag names throughout


def ensure_header():
    if not OUT_PATH.exists() or OUT_PATH.stat().st_size == 0:
        with open(OUT_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def poll_once() -> int:
    params = {"api_key": API_KEY, "boundingBox": BOUNDING_BOX}
    resp = requests.get(DATAFEED_URL, params=params, timeout=60)
    resp.raise_for_status()

    root = etree.parse(io.BytesIO(resp.content)).getroot()
    ns = detect_namespace(root)
    p = "siri:" if ns else ""  # xpath prefix; empty if file has no namespace

    activities = root.findall(f".//{p}VehicleActivity", ns)

    polled_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for act in activities:
        def text(tag):
            el = act.find(f".//{p}{tag}", ns)
            return el.text if el is not None else None

        rows.append({
            "polled_at_utc": polled_at,
            "recorded_at_time": text("RecordedAtTime"),
            "valid_until_time": text("ValidUntilTime"),
            "item_identifier": text("ItemIdentifier"),
            "vehicle_ref": text("VehicleRef"),
            "line_ref": text("LineRef"),
            "published_line_name": text("PublishedLineName"),
            "operator_ref": text("OperatorRef"),
            "direction_ref": text("DirectionRef"),
            "data_frame_ref": text("DataFrameRef"),
            "dated_vehicle_journey_ref": text("DatedVehicleJourneyRef"),
            "origin_ref": text("OriginRef"),
            "origin_name": text("OriginName"),
            "destination_ref": text("DestinationRef"),
            "destination_name": text("DestinationName"),
            "origin_aimed_departure_time": text("OriginAimedDepartureTime"),
            "longitude": text("Longitude"),
            "latitude": text("Latitude"),
            "bearing": text("Bearing"),
            "block_ref": text("BlockRef"),
            "vehicle_journey_ref": text("VehicleJourneyRef"),
        })

    if rows:
        with open(OUT_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true",
                         help="Keep polling forever instead of a single poll")
    parser.add_argument("--interval", type=int, default=30,
                         help="Seconds between polls in --loop mode (default 30)")
    args = parser.parse_args()

    ensure_header()

    if args.loop:
        print(f"Polling every {args.interval}s. Ctrl+C to stop. "
              f"Writing to {OUT_PATH}")
        try:
            while True:
                n = poll_once()
                print(f"{datetime.now().isoformat()} - polled {n} vehicle "
                      f"positions (file total: {sum(1 for _ in open(OUT_PATH)) - 1:,})")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped.")
    else:
        n = poll_once()
        print(f"Polled {n} vehicle positions -> {OUT_PATH}")