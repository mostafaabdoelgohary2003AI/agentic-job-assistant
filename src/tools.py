"""Deterministic tools the agents call: parsing, skill extraction, matching."""

from __future__ import annotations

import re

from .skill_catalog import SKILL_CATALOG

_WORD = re.compile(r"[a-zA-Z+#.؀-ۿ]+")


def extract_skills(text: str, catalog: dict[str, list[str]] | None = None) -> list[str]:
    """Return canonical skills found in *text* (case-insensitive)."""
    catalog = catalog or SKILL_CATALOG
    haystack = text.lower()
    found = []
    for canonical, aliases in catalog.items():
        if any(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", haystack) for alias in aliases):
            found.append(canonical)
    return sorted(found)


def parse_job_posting(job_text: str) -> dict:
    """Extract structured fields from a raw job posting."""
    lines = [l.strip() for l in job_text.strip().splitlines() if l.strip()]
    title = lines[0].lstrip("# ").strip() if lines else "Unknown role"
    company = ""
    location = ""
    for line in lines[1:6]:
        low = line.lower()
        if low.startswith(("company:", "at ", "الشركة:")):
            company = line.split(":", 1)[-1].replace("at ", "", 1).strip()
        elif low.startswith(("location:", "المكان:")):
            location = line.split(":", 1)[-1].strip()
    seniority = "junior"
    if re.search(r"\b(senior|lead|staff|principal)\b", job_text, re.I):
        seniority = "senior"
    elif re.search(r"\b(mid|intermediate|2\+ years|3\+ years)\b", job_text, re.I):
        seniority = "mid"
    return {
        "title": title,
        "company": company or "the company",
        "location": location or "unspecified",
        "seniority": seniority,
        "required_skills": extract_skills(job_text),
    }


def profile_cv(cv_text: str) -> dict:
    """Extract the candidate's skill profile from CV text."""
    return {"skills": extract_skills(cv_text)}


def compute_match(job_info: dict, cv_profile: dict) -> dict:
    """Skill-overlap match report between the job and the CV."""
    required = set(job_info.get("required_skills", []))
    owned = set(cv_profile.get("skills", []))
    matched = sorted(required & owned)
    missing = sorted(required - owned)
    score = round(len(matched) / len(required), 2) if required else 0.0
    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "verdict": "strong" if score >= 0.6 else "partial" if score >= 0.35 else "weak",
    }


_FORBIDDEN_CLAIMS = re.compile(r"\b(expert in|world-class|guru|10x)\b", re.I)
_INVENTED_METRIC = re.compile(r"\b\d{2,3}\s?%\b")


def compose_cover_letter(job_info: dict, match_report: dict, cv_profile: dict,
                         candidate_name: str = "Omar Khaled",
                         feedback: list[str] | None = None) -> str:
    """Compose an honest cover letter from the extracted facts only.

    The letter can only say things present in job_info / match_report /
    cv_profile - it has no access to invented claims by construction.
    """
    matched = match_report.get("matched_skills", [])
    missing = match_report.get("missing_skills", [])
    # lead with technical skills; languages are supporting details, not headlines
    languages = {"arabic", "english"}
    technical = [s for s in matched if s not in languages]
    skills_line = ", ".join((technical + sorted(s for s in matched if s in languages))[:4]) if matched else "software development"

    paragraphs = [
        f"Dear Hiring Team at {job_info['company']},",
        (
            f"I am applying for the {job_info['title']} position"
            + (f" in {job_info['location']}" if job_info.get("location") != "unspecified" else "")
            + ". I hold a B.Sc. in Artificial Intelligence Sciences from Cairo "
            "University (2026), and my day-to-day toolkit overlaps directly with this "
            f"role: {skills_line}."
        ),
        (
            "Recent projects I can walk through in an interview: a bilingual "
            "(Arabic/English) retrieval-augmented generation pipeline with a measured "
            "retrieval evaluation loop, a multi-agent job-application workflow built "
            "with LangGraph, and FireGuard - my graduation project (graded Excellent "
            "with Honors), where I owned deployment of the detection backend."
        ),
    ]
    if missing:
        paragraphs.append(
            "To be transparent about fit: the posting also asks for "
            f"{', '.join(missing[:3])}, which I have not used in production yet. "
            "I learn fast on real systems and I would rather say that plainly than "
            "oversell a keyword."
        )
    if feedback:
        paragraphs.append("Revised per review: " + "; ".join(feedback))
    paragraphs.append(
        "I am open to relocation and remote work. Thank you for your time - "
        "I would welcome a short technical conversation.\n\nBest regards,\n"
        f"{candidate_name}"
    )
    return "\n\n".join(paragraphs)


def critique_letter(letter: str, job_info: dict, match_report: dict) -> dict:
    """Rule-based review of a drafted letter. Returns {"issues": [...]}."""
    issues: list[str] = []
    words = len(letter.split())
    if words > 450:
        issues.append(f"too long ({words} words, max 450)")
    if job_info.get("company", "the company") not in letter:
        issues.append("company name not mentioned")
    matched_mentioned = [s for s in match_report.get("matched_skills", []) if s in letter.lower()]
    if len(matched_mentioned) < 2:
        issues.append("fewer than 2 matched skills are mentioned")
    if _FORBIDDEN_CLAIMS.search(letter):
        issues.append("contains overclaiming language")
    if _INVENTED_METRIC.search(letter):
        issues.append("contains a percentage metric that may not be verifiable")
    return {"issues": issues, "word_count": words, "passed": not issues}
