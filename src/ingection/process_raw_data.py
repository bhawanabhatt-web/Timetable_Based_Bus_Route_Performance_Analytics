
import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from lxml import etree
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def detect_namespace(root) -> dict:
    """Real BODS files may or may not declare a default namespace at the
    root. Detect it from the file itself rather than hardcoding a guess."""
    tag = root.tag
    if tag.startswith("{"):
        uri = tag[1:].split("}")[0]
        return {"ns": uri}
    return {}

def iter_xml_files(folder: Path):
    if not folder.exists():
        print(f"  (skipping, folder not found: {folder})")
        return
    yield from sorted(folder.rglob("*.xml"))

def write_csv(rows, out_path: Path, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> wrote {len(rows):,} rows to {out_path}")

# ---------------------------------------------------------------------------
# 1. TransXChange (Timetable) parser
# ---------------------------------------------------------------------------

def _text(el, path, ns, p):
    found = el.find(f"{p}{path}", ns)
    return found.text.strip() if found is not None and found.text else None


def parse_transxchange_file(path: Path):
    """Returns (stops, vehicle_journeys, stop_times) for a single TXC file."""
    stops, vjs, stop_times = [], [], []

    try:
        root = etree.parse(str(path)).getroot()
    except etree.XMLSyntaxError as e:
        print(f"  ! XML parse error in {path.name}: {e}")
        return stops, vjs, stop_times

    ns = detect_namespace(root)
    p = "ns:" if ns else ""
    source_file = path.name

    # --- Stops ---
    for sp in root.findall(f".//{p}AnnotatedStopPointRef", ns):
        stops.append({
            "source_file": source_file,
            "stop_point_ref": _text(sp, "StopPointRef", ns, p),
            "common_name": _text(sp, "CommonName", ns, p),
        })

    # --- Services (for operator/line context) ---
    service_lines = {}  # ServiceCode -> (line_name, description, operator_ref)
    for svc in root.findall(f".//{p}Services/{p}Service", ns):
        service_code = _text(svc, "ServiceCode", ns, p)
        description = _text(svc, "Description", ns, p)
        operator_ref = _text(svc, "RegisteredOperatorRef", ns, p)
        line_name = None
        line_el = svc.find(f".//{p}Lines/{p}Line/{p}LineName", ns)
        if line_el is not None:
            line_name = line_el.text
        service_lines[service_code] = (line_name, description, operator_ref)

    def parse_duration(dur):
        """Parses ISO-8601 durations like PT4M30S -> seconds. Returns 0 if
        missing (some links have no RunTime specified)."""
        if not dur:
            return 0
        dur = dur.replace("PT", "")
        seconds = 0
        num = ""
        for ch in dur:
            if ch.isdigit():
                num += ch
            elif ch == "H":
                seconds += int(num or 0) * 3600
                num = ""
            elif ch == "M":
                seconds += int(num or 0) * 60
                num = ""
            elif ch == "S":
                seconds += int(num or 0)
                num = ""
        return seconds

    jp_sections = {}
    for section in root.findall(f".//{p}JourneyPatternSections/{p}JourneyPatternSection", ns):
        section_id = section.get("id")
        links = []
        for link in section.findall(f"{p}JourneyPatternTimingLink", ns):
            from_stop = link.find(f"{p}From/{p}StopPointRef", ns)
            to_stop = link.find(f"{p}To/{p}StopPointRef", ns)
            run_time = _text(link, "RunTime", ns, p)
            links.append({
                "from_stop": from_stop.text if from_stop is not None else None,
                "to_stop": to_stop.text if to_stop is not None else None,
                "run_time_s": parse_duration(run_time),
            })
        jp_sections[section_id] = links

    # JourneyPatternRef -> ordered list of section ids
    jp_to_sections = {}
    for jp in root.findall(f".//{p}JourneyPattern", ns):
        jp_id = jp.get("id")
        section_refs = [el.text for el in jp.findall(f"{p}JourneyPatternSectionRefs", ns)]
        jp_to_sections[jp_id] = section_refs

    # --- Vehicle Journeys: the actual scheduled trips ---
    for vj in root.findall(f".//{p}VehicleJourneys/{p}VehicleJourney", ns):
        vj_code = _text(vj, "VehicleJourneyCode", ns, p)
        service_ref = _text(vj, "ServiceRef", ns, p)
        line_ref = _text(vj, "LineRef", ns, p)
        jp_ref = _text(vj, "JourneyPatternRef", ns, p)
        departure_time = _text(vj, "DepartureTime", ns, p)  # e.g. "07:30:00"

        line_name, description, operator_ref = service_lines.get(
            service_ref, (None, None, None)
        )

        vjs.append({
            "source_file": source_file,
            "vehicle_journey_code": vj_code,
            "service_ref": service_ref,
            "line_ref": line_ref,
            "line_name": line_name,
            "operator_ref": operator_ref,
            "journey_pattern_ref": jp_ref,
            "scheduled_departure_time": departure_time,
        })


        if not departure_time or jp_ref not in jp_to_sections:
            continue
        try:
            h, m, s = (int(x) for x in departure_time.split(":"))
            cumulative_s = h * 3600 + m * 60 + s
        except ValueError:
            continue

        stop_sequence = 0
        first_stop_written = False
        for section_id in jp_to_sections[jp_ref]:
            for link in jp_sections.get(section_id, []):
                if not first_stop_written:
                    stop_times.append({
                        "source_file": source_file,
                        "vehicle_journey_code": vj_code,
                        "line_ref": line_ref,
                        "stop_point_ref": link["from_stop"],
                        "stop_sequence": stop_sequence,
                        "scheduled_time": _seconds_to_hms(cumulative_s),
                    })
                    first_stop_written = True
                stop_sequence += 1
                cumulative_s += link["run_time_s"]
                stop_times.append({
                    "source_file": source_file,
                    "vehicle_journey_code": vj_code,
                    "line_ref": line_ref,
                    "stop_point_ref": link["to_stop"],
                    "stop_sequence": stop_sequence,
                    "scheduled_time": _seconds_to_hms(cumulative_s),
                })

    return stops, vjs, stop_times

def _seconds_to_hms(total_seconds):
    total_seconds = total_seconds % (24 * 3600)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def process_timetables(raw_dir: Path, out_dir: Path):
    print("\n[Timetable] parsing TransXChange files...")
    folder = raw_dir / "Timetable"
    all_stops, all_vjs, all_stop_times = [], [], []

    for xml_path in iter_xml_files(folder):
        stops, vjs, stop_times = parse_transxchange_file(xml_path)
        all_stops.extend(stops)
        all_vjs.extend(vjs)
        all_stop_times.extend(stop_times)

    write_csv(all_stops,out_dir / "timetable_stops.csv",
                ["source_file", "stop_point_ref", "common_name"])
    write_csv(all_vjs, out_dir / "timetable_vehicle_journeys.csv",
              ["source_file", "vehicle_journey_code", "service_ref", "line_ref",
               "line_name", "operator_ref", "journey_pattern_ref",
               "scheduled_departure_time"])
    write_csv(all_stop_times, out_dir / "timetable_stop_times.csv",
              ["source_file", "vehicle_journey_code", "line_ref",
               "stop_point_ref", "stop_sequence", "scheduled_time"])

# ---------------------------------------------------------------------------
# 2. SIRI-SX (Disruptions) parser
# ---------------------------------------------------------------------------

def process_disruptions(raw_dir: Path, out_dir: Path):
    print("\n[Disruptions] parsing SIRI-SX files...")
    folder = raw_dir / "Disruptions"
    rows = []

    for xml_path in iter_xml_files(folder):
        try:
            root = etree.parse(str(xml_path)).getroot()
        except etree.XMLSyntaxError as e:
            print(f"  ! XML parse error in {xml_path.name}: {e}")
            continue

        ns = detect_namespace(root)
        p = "ns:" if ns else ""

        for sit in root.findall(f".//{p}PtSituationElement", ns):
            rows.append({
                "source_file": xml_path.name,
                "situation_number": _text(sit, "SituationNumber", ns, p),
                "creation_time": _text(sit, "CreationTime", ns, p),
                "participant_ref": _text(sit, "ParticipantRef", ns, p),
                "version": _text(sit, "Version", ns, p),
                "progress": _text(sit, "Progress", ns, p),
                "misc_reason": _text(sit, "MiscellaneousReason", ns, p),
                "planned": _text(sit, "Planned", ns, p),
                "validity_start": _text(sit, "ValidityPeriod/StartTime", ns, p),
                "summary": _text(sit, "Summary", ns, p),
                "description": _text(sit, "Description", ns, p),
            })

    write_csv(rows, out_dir / "disruptions.csv",
              ["source_file", "situation_number", "creation_time",
               "participant_ref", "version", "progress", "misc_reason",
               "planned", "validity_start", "summary", "description"])

# ---------------------------------------------------------------------------
# 3. NeTEx (Fares) parser
# ---------------------------------------------------------------------------

def process_fares(raw_dir: Path, out_dir: Path):
    print("\n[Fares] parsing NeTEx files...")
    folder = raw_dir / "Fares"
    rows = []

    for xml_path in iter_xml_files(folder):
        try:
            root = etree.parse(str(xml_path)).getroot()
        except etree.XMLSyntaxError as e:
            print(f"  ! XML parse error in {xml_path.name}: {e}")
            continue

        ns = detect_namespace(root)
        p = "ns:" if ns else ""

        publication_timestamp = _text(root, "PublicationTimestamp", ns, p)
        participant_ref = _text(root, "ParticipantRef", ns, p)

        # OperatorRef / LineRef live inside NetworkFilterByValue/objectReferences
        operator_ref, line_ref, from_date, description = None, None, None, None
        op_el = root.find(f".//{p}NetworkFilterByValue//{p}OperatorRef", ns)
        if op_el is not None:
            operator_ref = op_el.get("ref")
        line_el = root.find(f".//{p}NetworkFilterByValue//{p}LineRef", ns)
        if line_el is not None:
            line_ref = line_el.get("ref")
        from_date_el = root.find(f".//{p}AvailabilityCondition/{p}FromDate", ns)
        if from_date_el is not None:
            from_date = from_date_el.text
        # Description appears more than once in a NeTEx file; take the one
        # inside PublicationRequest (a short human-readable summary).
        desc_el = root.find(f".//{p}PublicationRequest/{p}Description", ns)
        description = desc_el.text if desc_el is not None else None

        rows.append({
            "source_file": xml_path.name,
            "publication_timestamp": publication_timestamp,
            "participant_ref": participant_ref,
            "operator_ref": operator_ref,
            "line_ref": line_ref,
            "from_date": from_date,
            "description": description,
        })

    write_csv(rows, out_dir / "fares.csv",
              ["source_file", "publication_timestamp", "participant_ref",
               "operator_ref", "line_ref", "from_date", "description"])

# ---------------------------------------------------------------------------
# 4. Location (already CSV) -- just copy through
# ---------------------------------------------------------------------------

def process_location(raw_dir: Path, out_dir: Path):
    print("\n[Location] copying polled location CSV...")
    src = raw_dir / "Location" / "location_pings.csv"
    if not src.exists():
        print(f"  (skipping, not found: {src})")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "location_pings.csv"
    shutil.copyfile(src, dst)
    with open(src, encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1
    print(f"  -> copied {row_count:,} rows to {dst}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="Data/Raw Data")
    parser.add_argument("--out-dir", default="Data/Processed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    print(f"Processing raw data from: {raw_dir.resolve()}")
    print(f"Writing processed CSVs to: {out_dir.resolve()}")

    process_timetables(raw_dir, out_dir)
    process_disruptions(raw_dir, out_dir)
    process_fares(raw_dir, out_dir)
    process_location(raw_dir, out_dir)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()