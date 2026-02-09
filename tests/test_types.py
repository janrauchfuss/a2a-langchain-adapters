"""Tests for langchain_a2a_adapters.types."""

from langchain_a2a_adapters.types import A2AResult, A2AStreamEvent


class TestA2AResult:
    def test_minimal(self):
        r = A2AResult(task_id="t1", context_id="c1", status="completed")
        assert r.task_id == "t1"
        assert r.context_id == "c1"
        assert r.status == "completed"
        assert r.text is None
        assert r.data == []
        assert r.files == []
        assert r.artifacts == []
        assert r.requires_input is False

    def test_full(self):
        r = A2AResult(
            task_id="t1",
            context_id="c1",
            status="input-required",
            text="hello",
            data=[{"key": "value"}],
            files=[{"uri": "https://example.com/f.pdf"}],
            artifacts=[{"artifact_id": "a1", "name": "doc", "parts": []}],
            requires_input=True,
        )
        assert r.text == "hello"
        assert r.data == [{"key": "value"}]
        assert r.files[0]["uri"] == "https://example.com/f.pdf"
        assert r.artifacts[0]["artifact_id"] == "a1"
        assert r.requires_input is True

    def test_serialization_roundtrip(self):
        r = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            text="hi",
            data=[{"x": 1}],
        )
        d = r.model_dump()
        r2 = A2AResult.model_validate(d)
        assert r == r2


class TestA2AStreamEvent:
    def test_status_event(self):
        e = A2AStreamEvent(
            kind="status-update",
            task_id="t1",
            context_id="c1",
            status="working",
            final=False,
        )
        assert e.kind == "status-update"
        assert e.status == "working"
        assert e.final is False
        assert e.text is None
        assert e.data == []

    def test_artifact_event_with_text(self):
        e = A2AStreamEvent(
            kind="artifact-update",
            task_id="t1",
            context_id="c1",
            text="chunk of text",
        )
        assert e.kind == "artifact-update"
        assert e.text == "chunk of text"

    def test_artifact_event_with_data(self):
        e = A2AStreamEvent(
            kind="artifact-update",
            task_id="t1",
            context_id="c1",
            data=[{"result": 42}],
        )
        assert e.data == [{"result": 42}]
        assert e.text is None

    def test_final_event(self):
        e = A2AStreamEvent(
            kind="status-update",
            task_id="t1",
            context_id="c1",
            status="completed",
            final=True,
        )
        assert e.final is True

    def test_defaults(self):
        e = A2AStreamEvent(kind="status-update", task_id="t1", context_id="c1")
        assert e.text is None
        assert e.data == []
        assert e.status is None
        assert e.final is False
