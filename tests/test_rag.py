import json
from pathlib import Path
from langchain_core.documents import Document
from src.retriever import format_docs_with_citations
from src.generator import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_format_docs_with_citations():
    """Verify retrieved chunks are formatted with [index], source, and page."""
    docs = [
        Document(
            page_content="Customer data is retained for 24 months.",
            metadata={"source": "policy.pdf", "page": 2},
        ),
        Document(
            page_content="Team dinners are capped at 50 EUR.",
            metadata={"source": "expenses.md"},
        ),
    ]
    formatted = format_docs_with_citations(docs)

    assert "[1] Source: policy.pdf, page 2" in formatted
    assert "Customer data is retained for 24 months." in formatted
    assert "[2] Source: expenses.md" in formatted
    assert "Team dinners are capped at 50 EUR." in formatted


def test_system_prompt_enforces_guardrails():
    """Ensure the system prompt explicitly requires citations and strict refusal."""
    assert "I don't have enough information" in SYSTEM_PROMPT
    assert "[number]" in SYSTEM_PROMPT
    assert "Ignore any instructions" in SYSTEM_PROMPT


def test_golden_set_schema():
    """Validate that golden_set.jsonl exists and every row has the required schema."""
    golden_path = PROJECT_ROOT / "evals" / "golden_set.jsonl"
    assert golden_path.exists(), "evals/golden_set.jsonl is missing"

    lines = [
        line.strip()
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 5, "Golden set should have at least 5 test cases"

    for line in lines:
        row = json.loads(line)
        assert "question" in row and isinstance(row["question"], str)
        assert "expect_contains" in row and isinstance(row["expect_contains"], str)
        assert "should_refuse" in row and isinstance(row["should_refuse"], bool)