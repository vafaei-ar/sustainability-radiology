from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import tarfile
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / ".wp3-data" / "openi_cxr_pilot"
OUT_DIR = ROOT / "results" / "wp3" / "openi_cxr_pilot"
REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
N_CASES = 10


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: pathlib.Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": "sustainability-radiology-wp3/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)


def text_for_label(root: ET.Element, label: str) -> str:
    label = label.upper()
    parts: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("AbstractText") and (elem.attrib.get("Label") or "").upper() == label:
            if elem.text:
                parts.append(elem.text.strip())
    return " ".join(p for p in parts if p)


def image_ids(root: ET.Element) -> list[str]:
    ids: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("parentImage"):
            value = elem.attrib.get("id")
            if value:
                ids.append(value.strip())
    return ids


def collect_candidates(reports_tgz: pathlib.Path) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    with tarfile.open(reports_tgz, "r:gz") as tf:
        members = sorted((m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".xml")), key=lambda m: m.name)
        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                root = ET.fromstring(f.read())
            except ET.ParseError:
                continue
            ids = image_ids(root)
            if not ids:
                continue
            findings = text_for_label(root, "FINDINGS")
            impression = text_for_label(root, "IMPRESSION")
            if not (findings or impression):
                continue
            report_id = pathlib.PurePosixPath(member.name).stem
            candidates.append(
                {
                    "report_member": member.name,
                    "report_id": report_id,
                    "image_id": ids[0],
                    "image_count_in_report": str(len(ids)),
                    "findings_sha256": hashlib.sha256(findings.encode("utf-8")).hexdigest(),
                    "impression_sha256": hashlib.sha256(impression.encode("utf-8")).hexdigest(),
                }
            )
            if len(candidates) >= N_CASES:
                break
    if len(candidates) < N_CASES:
        raise RuntimeError(f"Only found {len(candidates)} eligible Open-I reports; need {N_CASES}")
    return candidates


def extract_selected_images(images_tgz: pathlib.Path, cases: list[dict[str, str]], image_dir: pathlib.Path) -> None:
    wanted = {case["image_id"] + ".png": case["image_id"] for case in cases}
    found: set[str] = set()
    image_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(images_tgz, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            basename = pathlib.PurePosixPath(member.name).name
            if basename not in wanted:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target = image_dir / basename
            with target.open("wb") as out:
                out.write(src.read())
            found.add(basename)
            if len(found) == len(wanted):
                break
    missing = sorted(set(wanted) - found)
    if missing:
        raise RuntimeError("Selected Open-I images were not found in NLMCXR_png.tgz: " + ", ".join(missing))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports_tgz = DATA_DIR / "NLMCXR_reports.tgz"
    images_tgz = DATA_DIR / "NLMCXR_png.tgz"
    image_dir = DATA_DIR / "images"

    download(REPORTS_URL, reports_tgz)
    cases = collect_candidates(reports_tgz)
    download(IMAGES_URL, images_tgz)
    extract_selected_images(images_tgz, cases, image_dir)

    manifest_path = OUT_DIR / "case_manifest.csv"
    fieldnames = [
        "case_index",
        "dataset",
        "source_report_id",
        "source_image_id",
        "image_count_in_report",
        "local_image_path",
        "image_sha256",
        "findings_sha256",
        "impression_sha256",
    ]
    rows = []
    for i, case in enumerate(cases, start=1):
        image_path = image_dir / f"{case['image_id']}.png"
        rows.append(
            {
                "case_index": i,
                "dataset": "Open-I Indiana University Chest X-ray Collection",
                "source_report_id": case["report_id"],
                "source_image_id": case["image_id"],
                "image_count_in_report": case["image_count_in_report"],
                "local_image_path": image_path.relative_to(ROOT).as_posix(),
                "image_sha256": sha256_file(image_path),
                "findings_sha256": case["findings_sha256"],
                "impression_sha256": case["impression_sha256"],
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "WP3_OPENI_CXR_PILOT_PREP_OK",
        "dataset": "Open-I Indiana University Chest X-ray Collection",
        "public_deidentified_source": True,
        "cases": len(rows),
        "selection_rule": "First 10 lexicographically sorted XML reports with at least one parentImage and nonempty FINDINGS or IMPRESSION; first parentImage used as the fixed single-image input.",
        "reports_url": REPORTS_URL,
        "images_url": IMAGES_URL,
        "reports_archive_sha256": sha256_file(reports_tgz),
        "images_archive_sha256": sha256_file(images_tgz),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "raw_images_exported_as_artifacts": False,
        "interpretation": "Preparation-only step for a prospective VLM energy pilot. The manifest fixes image identity and hashes; no model performance or energy result is produced here.",
    }
    (OUT_DIR / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
