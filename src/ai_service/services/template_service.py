import json
from importlib.resources import files
from typing import Any, Dict, List, Optional


class TemplateService:
    def __init__(self) -> None:
        self.templates_file = files("ai_service.data").joinpath("templates.json")
        self.templates_cache = None
        self.load_templates()

    def load_templates(self) -> None:
        """Загрузка шаблонов из JSON файла"""
        try:
            if self.templates_file.exists():
                with self.templates_file.open("r", encoding="utf-8") as f:
                    self.templates_cache = json.load(f)
                print(f"✅ Loaded {len(self.templates_cache.get('templates', []))} templates")
            else:
                print("⚠️ templates.json not found, using empty templates")
                self.templates_cache = {"templates": [], "categories": []}
        except Exception as e:
            print(f"❌ Error loading templates: {e}")
            self.templates_cache = {"templates": [], "categories": []}

    def get_all_templates(self) -> Dict[str, Any]:
        """Получить все шаблоны"""
        return self.templates_cache

    def get_template_by_id(self, template_id: str) -> Optional[Dict]:
        """Получить конкретный шаблон по ID"""
        templates = self.templates_cache.get("templates", [])
        for template in templates:
            if template["id"] == template_id:
                return template
        return None

    def get_templates_by_subject(self, subject: str) -> List[Dict]:
        """Получить шаблоны для конкретного предмета"""
        templates = self.templates_cache.get("templates", [])
        return [t for t in templates if subject in t.get("subjects", [])]

    def apply_template(self, template_id: str, user_input: str) -> str:
        """Применить шаблон с подстановкой user_input"""
        template = self.get_template_by_id(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        prompt = template["prompt"]

        for placeholder in template.get("placeholders", []):
            prompt = prompt.replace(f"{{{placeholder}}}", user_input)

        return prompt


template_service = TemplateService()
