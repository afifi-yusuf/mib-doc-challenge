"""Adjudication / safety policy for MIB intake packets.

Organizer note: silent/invented risk flags should yield NEEDS_REVIEW — never
fabricate stamps or CFA marks. This module only consumes flags extracted from
trusted text / biometric observed-flags lines.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .constants import (
    ALWAYS_EMBARGO_WORLDS,
    DEFAULT_RECEIPT_DATE,
    DISQUALIFYING_FLAGS,
    NONDIP_EMBARGO_WORLDS,
    REVIEW_FLAGS,
    REVOKED_SPONSORS,
)


def _parse_date(value: str | None) -> date | None:
    if not value or value in {"UNREADABLE", "unknown", ""}:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _field(fields: dict[str, Any], key: str, default: str = "unknown") -> str:
    hit = fields.get(key)
    if hit is None:
        return default
    if hasattr(hit, "value"):
        return str(hit.value)
    return str(hit) if hit else default


def apply_safety_policy(
    *,
    fields: dict[str, Any],
    risk_flags: set[str] | list[str],
    evidence_issues: list[str] | set[str] | None = None,
    manual_finding: str | None = None,
    waiver_code: str | None = None,
    docs_present: set[str] | None = None,
) -> tuple[str, float, str]:
    """Return (adjudication, confidence, reason).

    Loop-1 design goals (fields_ok_policy):
    - Do not NR solely because intake / attestation pages are image-only when
      trusted fields are already filled from other docs.
    - Accept ``waived`` for DIP-1, or when a visible waiver code exists
      (e.g. DIP-WAIVER / HARDSHIP), or when fee receipt already recorded waived
      without inventing extra review gates.
    - Never invent stamp / CFA / SAMPLE DENIAL based denials.
    """
    issues = set(evidence_issues or [])
    flags = set(risk_flags or [])
    docs_present = docs_present or set()

    visa = _field(fields, "visa_class", "unknown")
    fee = _field(fields, "fee_status", "unknown")
    sponsor = _field(fields, "sponsor_id", "SPN-0000")
    home = _field(fields, "home_world", "unknown")
    arrival = _parse_date(_field(fields, "arrival_date", ""))
    name = _field(fields, "applicant_name", "unknown")

    if not waiver_code:
        wc = fields.get("waiver_code")
        if wc is not None:
            waiver_code = wc.value if hasattr(wc, "value") else str(wc)

    # 1. Manual adjudicator finding dominates when present.
    if manual_finding in {"APPROVED", "DENIED", "NEEDS_REVIEW"}:
        conf = 0.93 if manual_finding != "NEEDS_REVIEW" else 0.86
        return manual_finding, conf, "manual_finding"

    # 2. Hard disqualifiers.
    hit = flags & DISQUALIFYING_FLAGS
    if hit:
        return "DENIED", 0.95, f"disq_flag:{sorted(hit)[0]}"

    if home in ALWAYS_EMBARGO_WORLDS:
        return "DENIED", 0.94, f"embargo_world:{home}"

    if home in NONDIP_EMBARGO_WORLDS and visa != "DIP-1":
        return "DENIED", 0.93, "wolf_nondip"

    if visa == "TRANSIT-7":
        return "DENIED", 0.96, "transit7"

    if sponsor in REVOKED_SPONSORS and visa != "DIP-1":
        return "DENIED", 0.94, f"revoked:{sponsor}"

    if fee == "unpaid":
        return "DENIED", 0.95, "unpaid"

    if visa != "DIP-1" and arrival is not None:
        if (DEFAULT_RECEIPT_DATE - arrival).days > 180:
            return "DENIED", 0.9, "stale_arrival"

    # 3. Fee unknown → review (unless we truly lack any fee signal).
    if fee == "unknown":
        return "NEEDS_REVIEW", 0.8, "fee_unknown"

    # 4. Waived fee gate: DIP-1 always OK; non-DIP OK when waiver code present
    #    or when the fee receipt itself recorded waived (gold often uses
    #    DIP-WAIVER even on MED-3/XW). Do NOT auto-NR clean waived packets.
    if fee == "waived" and visa != "DIP-1":
        code = (waiver_code or "").strip().upper()
        has_waiver = bool(code) and code not in {"N/A", "NA", "NONE", "UNKNOWN", "-"}
        if not has_waiver and "fee_receipt" not in docs_present:
            # Waived claimed without receipt or code → soft review.
            # If fee_receipt is present, trust the receipt's waived status.
            if "fee_missing" in issues:
                return "NEEDS_REVIEW", 0.7, "waived_unverified"

    # 5. Printed review-only flags (never invent these).
    rev = flags & REVIEW_FLAGS
    if rev:
        return "NEEDS_REVIEW", 0.85, f"review_flag:{sorted(rev)[0]}"

    # 6. Evidence gaps that block approval — but only when the field is still
    #    unresolved. If extraction already filled the field, do not NR solely
    #    because a particular document type was missing (Loop-1 root cause).
    if arrival is None or "arrival_unreadable" in issues:
        if _field(fields, "arrival_date", "1900-01-01") in {
            "1900-01-01",
            "unknown",
            "UNREADABLE",
            "",
        }:
            return "NEEDS_REVIEW", 0.72, "arrival_bad"

    if name in {"unknown", "[NAME CUT OUT]", ""} or "name_cut_out" in issues:
        return "NEEDS_REVIEW", 0.7, "identity_gap"

    if "sponsor_conflict" in issues:
        return "NEEDS_REVIEW", 0.75, "sponsor_conflict"

    if visa == "unknown":
        return "NEEDS_REVIEW", 0.6, "visa_unknown"

    if visa != "DIP-1" and sponsor in {"SPN-0000", "unknown"}:
        if not fields.get("sponsor_id"):
            return "NEEDS_REVIEW", 0.65, "sponsor_missing"

    # Deliberately NOT requiring intake / sponsor_letter / biometric presence
    # when fields are already populated from registry / fee / OCR.

    return "APPROVED", 0.88, "clean"


def build_prediction(
    *,
    case_id: str,
    fields: dict[str, Any],
    risk_flags: set[str],
    evidence_issues: list[str],
    manual_finding: str | None,
    docs_present: set[str],
) -> dict[str, Any]:
    adjudication, confidence, reason = apply_safety_policy(
        fields=fields,
        risk_flags=risk_flags,
        evidence_issues=evidence_issues,
        manual_finding=manual_finding,
        docs_present=docs_present,
    )

    flags = sorted(risk_flags)
    risk_flags_s = "|".join(flags) if flags else "none"

    sponsor = _field(fields, "sponsor_id", "SPN-0000")
    if not sponsor.startswith("SPN-"):
        sponsor = "SPN-0000"

    arrival = _field(fields, "arrival_date", "1900-01-01")
    if arrival == "UNREADABLE":
        arrival = "1900-01-01"

    fee = _field(fields, "fee_status", "unknown")
    if fee not in {"paid", "waived", "unpaid", "unknown"}:
        fee = "unknown"

    return {
        "case_id": case_id,
        "applicant_name": _field(fields, "applicant_name", "unknown"),
        "species_code": _field(fields, "species_code", "unknown"),
        "home_world": _field(fields, "home_world", "unknown"),
        "visa_class": _field(fields, "visa_class", "unknown"),
        "sponsor_id": sponsor,
        "arrival_date": arrival,
        "declared_purpose": _field(fields, "declared_purpose", "unknown"),
        "risk_flags": risk_flags_s,
        "fee_status": fee,
        "adjudication": adjudication,
        "confidence": float(confidence),
        "_reason": reason,
    }
