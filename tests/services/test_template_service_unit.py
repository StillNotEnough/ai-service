import pytest

from ai_service.services.template_service import TemplateService


def _service_with_cache(cache: dict) -> TemplateService:
    """
    Создаём TemplateService и подменяем его cache,
    чтобы тесты не зависели от templates.json.
    """
    svc = TemplateService()
    svc.templates_cache = cache
    return svc


def test_get_template_by_id_returns_template_when_exists():
    svc = _service_with_cache(
        {
            "templates": [
                {
                    "id": "t1",
                    "prompt": "Explain: {topic}",
                    "placeholders": ["topic"],
                    "subjects": ["general"],
                }
            ],
            "categories": [],
        }
    )

    t = svc.get_template_by_id("t1")
    assert t is not None
    assert t["id"] == "t1"


def test_get_template_by_id_returns_none_when_missing():
    svc = _service_with_cache({"templates": [], "categories": []})
    assert svc.get_template_by_id("missing") is None


def test_apply_template_replaces_placeholders():
    svc = _service_with_cache(
        {
            "templates": [
                {
                    "id": "t1",
                    "prompt": "Explain: {topic}",
                    "placeholders": ["topic"],
                    "subjects": ["general"],
                }
            ],
            "categories": [],
        }
    )

    out = svc.apply_template("t1", "Newton's 3rd law")
    assert out == "Explain: Newton's 3rd law"


def test_apply_template_raises_value_error_when_template_not_found():
    svc = _service_with_cache({"templates": [], "categories": []})
    with pytest.raises(ValueError):
        svc.apply_template("nope", "x")
