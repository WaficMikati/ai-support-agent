"""Tests for the loop and the startup wiring.

Two things here that only bite in front of an audience: editing the knowledge
file having no effect until a restart, and a missing variable crashing with a
bare KeyError.
"""

import pytest

from agent import Classification, Conversation, Message, load_env_file, require_env, run_forever


class StopLoop(Exception):
    pass


class RecordingInbox:
    """Answers once per pass and records the knowledge it was given."""

    def __init__(self):
        self.replies: list[str] = []
        self.resolved: list[int] = []

    def open_conversations(self):
        return [
            Conversation(
                id=1,
                contact_email="a@b.com",
                messages=(Message(id=1, content="hello", incoming=True),),
            )
        ]

    def send_reply(self, conversation_id, content):
        self.replies.append(content)

    def add_private_note(self, conversation_id, content):
        pass

    def resolve(self, conversation_id):
        self.resolved.append(conversation_id)


def run_passes(knowledge_file, passes: int, inbox) -> None:
    """Drive run_forever for a fixed number of passes by raising from sleep."""
    counter = {"n": 0}

    def sleep(_seconds):
        counter["n"] += 1
        if counter["n"] >= passes:
            raise StopLoop

    with pytest.raises(StopLoop):
        run_forever(
            knowledge_path=knowledge_file,
            sleep=sleep,
            inbox=inbox,
            payments=None,
            classify=lambda message: Classification("support", 0.99),
            answer=lambda message, knowledge: knowledge.strip(),
            interval_seconds=0,
        )


# ------------------------------------------------------------ live reload


def test_editing_the_knowledge_file_takes_effect_without_a_restart(tmp_path):
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("first guidance")
    inbox = RecordingInbox()

    counter = {"n": 0}

    def sleep(_seconds):
        counter["n"] += 1
        if counter["n"] == 1:
            knowledge.write_text("second guidance")  # edited while running
        if counter["n"] >= 2:
            raise StopLoop

    with pytest.raises(StopLoop):
        run_forever(
            knowledge_path=knowledge,
            sleep=sleep,
            inbox=inbox,
            payments=None,
            classify=lambda message: Classification("support", 0.99),
            answer=lambda message, knowledge_text: knowledge_text.strip(),
            interval_seconds=0,
        )

    assert inbox.replies == ["first guidance", "second guidance"]


def test_a_failing_pass_does_not_kill_the_loop(tmp_path):
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("guidance")

    class ExplodingInbox(RecordingInbox):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def open_conversations(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("chatwoot down")
            return super().open_conversations()

    inbox = ExplodingInbox()
    run_passes(knowledge, 3, inbox)
    assert inbox.calls >= 2, "the loop kept going after the failure"
    assert inbox.replies, "and recovered enough to answer"


# ---------------------------------------------------------------- env file


def test_env_file_is_loaded(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\nFOO_ONE=alpha\n\nFOO_TWO = beta \n")
    monkeypatch.delenv("FOO_ONE", raising=False)
    monkeypatch.delenv("FOO_TWO", raising=False)
    load_env_file(env)
    assert require_env("FOO_ONE") == "alpha"
    assert require_env("FOO_TWO") == "beta"


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_THREE=from-file\n")
    monkeypatch.setenv("FOO_THREE", "from-shell")
    load_env_file(env)
    assert require_env("FOO_THREE") == "from-shell"


def test_a_missing_env_file_is_not_an_error(tmp_path):
    load_env_file(tmp_path / "nope.env")  # must not raise


def test_missing_variable_explains_itself(monkeypatch):
    monkeypatch.delenv("FOO_MISSING", raising=False)
    with pytest.raises(SystemExit) as failure:
        require_env("FOO_MISSING", "Get it from somewhere.")
    message = str(failure.value)
    assert "FOO_MISSING" in message
    assert ".env" in message
    assert "Get it from somewhere." in message


def test_blank_variable_counts_as_missing(monkeypatch):
    monkeypatch.setenv("FOO_BLANK", "   ")
    with pytest.raises(SystemExit):
        require_env("FOO_BLANK")
