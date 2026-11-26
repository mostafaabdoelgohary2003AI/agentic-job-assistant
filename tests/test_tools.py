import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools import compute_match, critique_letter, extract_skills, parse_job_posting, profile_cv

JOB = Path(__file__).resolve().parents[1].joinpath("examples/sample_job.txt").read_text(encoding="utf-8")
CV = Path(__file__).resolve().parents[1].joinpath("examples/sample_cv.txt").read_text(encoding="utf-8")


def test_extract_skills_finds_canonical_names():
    skills = extract_skills("We use Python, PyTorch and retrieval-augmented generation.")
    assert "python" in skills and "pytorch" in skills and "rag" in skills


def test_parse_job_posting_fields():
    info = parse_job_posting(JOB)
    assert info["title"] == "Junior AI Engineer"
    assert info["seniority"] == "junior"
    assert "python" in info["required_skills"] and "rag" in info["required_skills"]


def test_match_report_scores_overlap():
    report = compute_match(parse_job_posting(JOB), profile_cv(CV))
    assert report["score"] >= 0.5, report
    assert report["verdict"] in {"strong", "partial"}
    assert set(report["matched_skills"]).isdisjoint(report["missing_skills"])


def test_critique_catches_overclaim():
    bad = "I am an expert in everything. I improved accuracy by 99%. " * 30
    result = critique_letter(bad, parse_job_posting(JOB), {"matched_skills": []})
    assert not result["passed"]
    assert any("overclaiming" in i or "metric" in i for i in result["issues"])


if __name__ == "__main__":
    test_extract_skills_finds_canonical_names()
    test_parse_job_posting_fields()
    test_match_report_scores_overlap()
    test_critique_catches_overclaim()
    print("tools tests passed")
