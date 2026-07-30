"""Tests for the loop and the startup wiring.

Two things here that only bite in front of an audience: editing the knowledge
file having no effect until a restart, and a missing variable crashing with a
bare KeyError.
"""

import pytest

from agent import (
    Conversation,
    Message,
    Proposal,
    env_value,
    load_env_file,
    require_env,
    run_forever,
)


class StopLoop(Exception):
    pass


class RecordingInbox:
    """A new customer message on every pass, recording the knowledge used.

    The message id increments because the agent now remembers what it has
    already handled, so re-serving an identical message would correctly be
    ignored rather than answered again.
    """

    def __init__(self):
        self.replies: list[str] = []
        self.resolved: list[int] = []
        self.recorded: list[tuple[int, int]] = []
        self._next_id = 0

    def open_conversations(self):
        self._next_id += 1
        return [
            Conversation(
                id=1,
                contact_email="a@b.com",
                messages=(
                    Message(id=self._next_id, content="hello", incoming=True),
                ),
            )
        ]

    def send_reply(self, conversation_id, content):
        self.replies.append(content)

    def add_private_note(self, conversation_id, content):
        pass

    def resolve(self, conversation_id):
        self.resolved.append(conversation_id)

    def record_handled(self, conversation_id, message_id):
        pass

    def record_handled(self, conversation_id, message_id):
        self.recorded.append((conversation_id, message_id))


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
            understand=lambda turns, knowledge, articles=(): Proposal(
                reply=knowledge.strip(),
                refund_requested=False,
                clear_request=True,
                charge_identified=True,
                hedging=False,
            ),
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
            understand=lambda turns, knowledge_text, articles=(): Proposal(
                reply=knowledge_text.strip(),
                refund_requested=False,
                clear_request=True,
                charge_identified=True,
                hedging=False,
            ),
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
        require_env("FOO_MISSING", hint="Get it from somewhere.")
    message = str(failure.value)
    assert "FOO_MISSING" in message
    assert ".env" in message
    assert "Get it from somewhere." in message


def test_blank_variable_counts_as_missing(monkeypatch):
    monkeypatch.setenv("FOO_BLANK", "   ")
    with pytest.raises(SystemExit):
        require_env("FOO_BLANK")


# ------------------------------------------------- provider-neutral env names


def test_the_model_name_is_preferred_over_the_openrouter_one(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "new")
    monkeypatch.setenv("OPENROUTER_API_KEY", "old")
    assert env_value("MODEL_API_KEY", "OPENROUTER_API_KEY") == "new"


def test_the_openrouter_name_still_works(monkeypatch):
    """Existing .env files must keep running after the rename."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "old")
    assert env_value("MODEL_API_KEY", "OPENROUTER_API_KEY") == "old"


def test_a_default_is_used_when_neither_is_set(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert env_value("MODEL_NAME", "OPENROUTER_MODEL", default="fallback") == "fallback"


def test_a_blank_value_falls_through(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "   ")
    monkeypatch.setenv("OPENROUTER_MODEL", "real")
    assert env_value("MODEL_NAME", "OPENROUTER_MODEL") == "real"
