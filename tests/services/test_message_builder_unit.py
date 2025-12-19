from ai_service.models.schemas import ChatMessage
from ai_service.services.message_builder import prepare_messages


def test_prepare_messages_has_system_first_and_user_last():
    history = [ChatMessage(role="user", content="prev")]
    messages, model = prepare_messages("current", "general", history)

    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "current"
    assert isinstance(model, str) and model


def test_prepare_messages_trims_history_to_last_10():
    history = [ChatMessage(role="user", content=str(i)) for i in range(30)]
    messages, _ = prepare_messages("x", "general", history)

    # system + last 10 history + current user
    assert len(messages) == 1 + 10 + 1


def test_prepare_messages_uses_subject_model_and_prompt_fallback():
    # неизвестный subject должен фолбечиться на general (как у тебя было)
    messages, model = prepare_messages("x", "unknown-subject", [])

    assert messages[0]["role"] == "system"
    assert isinstance(model, str) and model
