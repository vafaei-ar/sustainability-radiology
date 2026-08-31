from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "wp4" / "cms_imaging_volume"
CACHE = ROOT / "data" / "wp4" / "cms_imaging_volume"
CATALOG_URL = "https://data.cms.gov/data.json"
CMS_TITLE = "Medicare Physician & Other Practitioners - by Geography and Service"
RBCS_TITLE = "Restructured BETOS Classification System"
TARGET_YEAR = 2024
PAGE_SIZE = 5000
USER_AGENT = "sustainability-radiology/1.0"


def progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total if total else None,
        "phase": phase,
        "unit": "CMS validation stages",
        "updated_at_epoch": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def fetch_json(url: str, timeout: int = 120) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_catalog() -> tuple[dict, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    (CACHE / "data.json").write_bytes(data)
    return json.loads(data), sha256_bytes(data)


def catalog_dataset(catalog: dict, title: str) -> dict:
    candidates = [d for d in catalog.get("dataset", []) if str(d.get("title", "")).strip() == title]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one catalog dataset titled {title!r}; found {len(candidates)}")
    return candidates[0]


def year_from_distribution(d: dict) -> int | None:
    temporal = str(d.get("temporal", ""))
    m = re.match(r"(\d{4})-", temporal)
    if m:
        return int(m.group(1))
    title = str(d.get("title", ""))
    years = [int(x) for x in re.findall(r"\b(20\d{2})\b", title)]
    return max(years) if years else None


def choose_api_distribution(ds: dict, preferred_year: int | None = None) -> dict:
    api = [d for d in ds.get("distribution", []) if str(d.get("format", "")).upper() == "API" and d.get("accessURL")]
    if not api:
        raise RuntimeError(f"No API distribution for {ds.get('title')}")
    if preferred_year is not None:
        same = [d for d in api if year_from_distribution(d) == preferred_year]
        if same:
            api = same
    api.sort(key=lambda d: (str(d.get("modified", "")), str(d.get("title", ""))), reverse=True)
    latest = [d for d in api if str(d.get("description", "")).lower() == "latest"]
    return latest[0] if latest else api[0]


def api_page(access_url: str, offset: int, filters: dict[str, str] | None = None) -> list[dict]:
    params: list[tuple[str, str]] = [("size", str(PAGE_SIZE)), ("offset", str(offset))]
    for key, value in (filters or {}).items():
        params.append((f"filter[{key}]", value))
    url = access_url + ("&" if "?" in access_url else "?") + urllib.parse.urlencode(params)
    obj = fetch_json(url, timeout=180)
    if not isinstance(obj, list):
        raise RuntimeError(f"Unexpected CMS API response type at offset {offset}: {type(obj).__name__}")
    return obj


def fetch_all(access_url: str, filters: dict[str, str] | None = None, max_pages: int = 500) -> list[dict]:
    rows: list[dict] = []
    for page in range(max_pages):
        batch = api_page(access_url, page * PAGE_SIZE, filters)
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        time.sleep(0.05)
    raise RuntimeError("CMS API pagination exceeded safety page limit")


def nkey(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def find_field(row: dict, exact: tuple[str, ...] = (), contains: tuple[str, ...] = ()) -> str:
    norm = {nkey(k): k for k in row}
    for candidate in exact:
        if nkey(candidate) in norm:
            return norm[nkey(candidate)]
    for key in row:
        nk = nkey(key)
        if all(nkey(piece) in nk for piece in contains):
            return key
    raise RuntimeError(f"Could not identify field exact={exact} contains={contains}; fields={list(row)}")


def optional_field(row: dict, exact: tuple[str, ...], contains: tuple[str, ...] = ()) -> str | None:
    try:
        return find_field(row, exact=exact, contains=contains)
    except RuntimeError:
        return None


def as_float(value: object) -> float:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def classify_modality(text: str) -> str:
    t = text.lower()
    if re.search(r"mamm|breast.*imag", t):
        return "mammography"
    if re.search(r"magnetic resonance|\bmri\b|\bmra\b|\bmr\b", t):
        return "mri"
    if re.search(r"computed tomography|\bct\b|\bcta\b|cat scan", t):
        return "ct"
    if re.search(r"pet|positron|nuclear|scint|spect", t):
        return "nuclear_medicine_pet"
    if re.search(r"ultrasound|ultrason|sonograph|echograph|duplex", t):
        return "ultrasound"
    if re.search(r"x[- ]?ray|radiograph|plain film|fluorosc|angiograph", t):
        return "radiography_fluoroscopy"
    return "other_imaging"


def write_csv(path: pathlib.Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    progress(0, 5, "Downloading CMS catalog")
    catalog, catalog_sha = download_catalog()

    cms_ds = catalog_dataset(catalog, CMS_TITLE)
    cms_dist = choose_api_distribution(cms_ds, TARGET_YEAR)
    cms_url = str(cms_dist["accessURL"])
    sample = api_page(cms_url, 0)
    if not sample:
        raise RuntimeError("CMS geography/service API returned no rows")
    geo_field = find_field(sample[0], exact=("Rndrng_Prvdr_Geo_Lvl",), contains=("GEO", "LVL"))
    hcpcs_field = find_field(sample[0], exact=("HCPCS_Cd",), contains=("HCPCS", "CD"))
    desc_field = find_field(sample[0], exact=("HCPCS_Desc",), contains=("HCPCS", "DESC"))
    service_field = find_field(sample[0], exact=("Tot_Srvcs",), contains=("TOT", "SRVC"))
    progress(1, 5, "Resolved 2024 CMS geography/service schema")

    national = fetch_all(cms_url, {geo_field: "National"})
    if not national:
        national = fetch_all(cms_url, {geo_field: "NATIONAL"})
    if not national:
        raise RuntimeError("No national rows found in CMS geography/service dataset")
    progress(2, 5, f"Downloaded {len(national)} national CMS service rows")

    rbcs_ds = catalog_dataset(catalog, RBCS_TITLE)
    rbcs_dist = choose_api_distribution(rbcs_ds)
    rbcs_url = str(rbcs_dist["accessURL"])
    rbcs = fetch_all(rbcs_url)
    if not rbcs:
        raise RuntimeError("RBCS API returned no rows")

    first = rbcs[0]
    r_hcpcs = find_field(first, exact=("HCPCS_CD", "HCPCS_Cd"), contains=("HCPCS", "CD"))
    r_cat_code = optional_field(first, exact=("RBCS_CAT",), contains=("RBCS", "CAT"))
    r_cat_desc = optional_field(first, exact=("RBCS_CAT_DESC", "RBCS_CATEGORY_DESC"), contains=("RBCS", "CAT", "DESC"))
    r_subcat_code = optional_field(first, exact=("RBCS_SUBCAT",), contains=("RBCS", "SUBCAT"))
    r_subcat_desc = optional_field(first, exact=("RBCS_SUBCAT_DESC", "RBCS_SUBCATEGORY_DESC"), contains=("RBCS", "SUBCAT", "DESC"))
    r_family_code = optional_field(first, exact=("RBCS_FAMILY",), contains=("RBCS", "FAMILY"))
    r_family_desc = optional_field(first, exact=("RBCS_FAMILY_DESC",), contains=("RBCS", "FAMILY", "DESC"))

    if r_cat_code is None and r_cat_desc is None:
        raise RuntimeError(f"Could not identify RBCS category fields; fields={list(first)}")

    rbcs_map: dict[str, dict] = {}
    for row in rbcs:
        code = str(row.get(r_hcpcs, "")).strip().upper()
        if not code:
            continue
        category_code = str(row.get(r_cat_code, "")).strip() if r_cat_code else ""
        category_desc = str(row.get(r_cat_desc, "")).strip() if r_cat_desc else ""
        is_imaging = category_code.upper() == "I" or "IMAGING" in category_desc.upper()
        if not is_imaging:
            continue
        subcat_code = str(row.get(r_subcat_code, "")).strip() if r_subcat_code else ""
        subcat_desc = str(row.get(r_subcat_desc, "")).strip() if r_subcat_desc else ""
        family_code = str(row.get(r_family_code, "")).strip() if r_family_code else ""
        family_desc = str(row.get(r_family_desc, "")).strip() if r_family_desc else ""
        rbcs_map[code] = {
            "category": category_desc or category_code or "Imaging",
            "subcategory": subcat_desc or subcat_code,
            "family": family_desc or family_code,
            "category_code": category_code,
            "subcategory_code": subcat_code,
            "family_code": family_code,
        }
    if not rbcs_map:
        raise RuntimeError(f"No RBCS Imaging mappings found; detected category fields code={r_cat_code!r}, desc={r_cat_desc!r}")
    progress(3, 5, f"Resolved {len(rbcs_map)} RBCS Imaging HCPCS codes")

    mapped_rows: list[dict] = []
    unmatched_services = 0.0
    total_services = 0.0
    for row in national:
        code = str(row.get(hcpcs_field, "")).strip().upper()
        services = as_float(row.get(service_field))
        total_services += services
        info = rbcs_map.get(code)
        if info is None:
            unmatched_services += services
            continue
        desc = str(row.get(desc_field, "")).strip()
        combined = " | ".join([desc, info["subcategory"], info["family"]])
        mapped_rows.append({
            "hcpcs_code": code,
            "hcpcs_description": desc,
            "rbcs_category": info["category"],
            "rbcs_subcategory": info["subcategory"],
            "rbcs_family": info["family"],
            "derived_modality": classify_modality(combined),
            "total_services": services,
        })

    if not mapped_rows:
        raise RuntimeError("No national CMS rows matched RBCS Imaging codes")

    by_sub: dict[tuple[str, str], float] = {}
    by_mod: dict[str, float] = {}
    imaging_total = 0.0
    for r in mapped_rows:
        imaging_total += float(r["total_services"])
        key = (r["rbcs_subcategory"], r["rbcs_family"])
        by_sub[key] = by_sub.get(key, 0.0) + float(r["total_services"])
        by_mod[r["derived_modality"]] = by_mod.get(r["derived_modality"], 0.0) + float(r["total_services"])

    sub_rows = [
        {"rbcs_subcategory": k[0], "rbcs_family": k[1], "total_services": f"{v:.3f}", "share_of_imaging_services": f"{v / imaging_total:.9f}"}
        for k, v in sorted(by_sub.items(), key=lambda kv: kv[1], reverse=True)
    ]
    mod_rows = [
        {"derived_modality": k, "total_services": f"{v:.3f}", "share_of_imaging_services": f"{v / imaging_total:.9f}"}
        for k, v in sorted(by_mod.items(), key=lambda kv: kv[1], reverse=True)
    ]
    mapped_rows.sort(key=lambda r: float(r["total_services"]), reverse=True)

    write_csv(OUT / "national_imaging_by_rbcs.csv", ["rbcs_subcategory", "rbcs_family", "total_services", "share_of_imaging_services"], sub_rows)
    write_csv(OUT / "national_imaging_by_modality.csv", ["derived_modality", "total_services", "share_of_imaging_services"], mod_rows)
    write_csv(OUT / "hcpcs_imaging_mapping_audit.csv", ["hcpcs_code", "hcpcs_description", "rbcs_category", "rbcs_subcategory", "rbcs_family", "derived_modality", "total_services"], mapped_rows)

    summary = {
        "status": "WP4_CMS_IMAGING_VOLUME_OK",
        "cms_dataset": CMS_TITLE,
        "cms_year": year_from_distribution(cms_dist),
        "cms_distribution_title": cms_dist.get("title"),
        "cms_api": cms_url,
        "rbcs_dataset": RBCS_TITLE,
        "rbcs_distribution_title": rbcs_dist.get("title"),
        "rbcs_api": rbcs_url,
        "catalog_sha256": catalog_sha,
        "national_rows_downloaded": len(national),
        "rbcs_imaging_codes": len(rbcs_map),
        "matched_imaging_hcpcs_rows": len(mapped_rows),
        "imaging_total_services": imaging_total,
        "all_national_service_total_before_rbcs_filter": total_services,
        "unmatched_nonimaging_or_unmapped_service_total": unmatched_services,
        "modality_totals": {k: v for k, v in sorted(by_mod.items())},
        "rbcs_fields": {
            "category_code": r_cat_code,
            "category_description": r_cat_desc,
            "subcategory_code": r_subcat_code,
            "subcategory_description": r_subcat_desc,
            "family_code": r_family_code,
            "family_description": r_family_desc,
        },
        "interpretation": "CMS Original Medicare fee-for-service Part B service counts are an external utilization validation layer, not full-US examination counts.",
        "limitations": [
            "Service counts are billing services and may not equal unique imaging examinations; professional and technical billing structure can affect counts.",
            "The dataset represents Original Medicare fee-for-service Part B, not Medicare Advantage or the full US population.",
            "CMS suppresses/redacts selected low-count information for beneficiary privacy.",
            "Derived modality labels are transparent text-based groupings of RBCS Imaging codes and should be treated as analytic categories, not clinical adjudication.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# WP4 CMS imaging-volume validation

Status: PASS

Source: CMS `{CMS_TITLE}`, data year {summary['cms_year']}, joined to the latest available `{RBCS_TITLE}` taxonomy discovered from the CMS data catalog at execution time.

National CMS rows downloaded: {len(national):,}. RBCS Imaging HCPCS codes: {len(rbcs_map):,}. Matched imaging HCPCS rows: {len(mapped_rows):,}. Total matched imaging service count: {imaging_total:,.0f}.

This output is an **external validation layer**. It is not a national US examination denominator. CMS service counts reflect Original Medicare fee-for-service Part B billing services and may differ from unique examinations because of billing components and service definitions. Medicare Advantage and non-Medicare populations are not represented.

RBCS identifies Imaging with category code `I`; category/subcategory/family description fields are used when available. The modality table is derived from RBCS Imaging taxonomy text plus HCPCS descriptions and is intended for broad workload-distribution checks. The RBCS subcategory/family table is retained as the less transformed primary CMS summary.

Raw CMS public data are cached project-locally and are not declared as RunRelay artifacts. Only derived aggregate/audit tables are exported.
"""
    (OUT / "validation_report.md").write_text(report, encoding="utf-8")
    progress(5, 5, "CMS imaging-volume validation complete")
    print("WP4_CMS_IMAGING_VOLUME_OK")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
