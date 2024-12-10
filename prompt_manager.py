import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class Prompt:
    text: str
    negative: str
    category: str
    subcategory: str

class PromptLibrary:
    def __init__(self, library_file: Path):
        self.prompts: Dict[str, List[Prompt]] = {}
        self.categories: Dict[str, str] = {}  # id -> display name
        self.subcategories: Dict[str, Dict[str, str]] = {}  # category -> {id -> display name}
        self._load_library(library_file)

    def _load_library(self, library_file: Path):
        """Load prompts from JSON file."""
        with open(library_file) as f:
            data = json.load(f)
            
        # Process each category
        for cat_id, cat_data in data.items():
            self.categories[cat_id] = cat_data['name']
            self.subcategories[cat_id] = {}
            self.prompts[cat_id] = []
            
            # Process subcategories
            for subcat_id, subcat_data in cat_data['subcategories'].items():
                self.subcategories[cat_id][subcat_id] = subcat_data['name']
                
                # Process prompts
                for prompt in subcat_data['prompts']:
                    self.prompts[cat_id].append(
                        Prompt(
                            text=prompt['text'],
                            negative=prompt['negative'],
                            category=cat_id,
                            subcategory=subcat_id
                        )
                    )

    def get_categories(self) -> List[Tuple[str, str]]:
        """Get list of category id/name pairs."""
        return list(self.categories.items())

    def get_subcategories(self, category) -> List[Tuple[str, str]]:
        """Get subcategories for a category as id/name pairs."""
        if isinstance(category, dict):
            category = category["value"]
        elif not category:
            return []
            
        subcats = self.subcategories.get(category, {})
        return list(subcats.items())

    def get_prompts(self, category: str, subcategory: Optional[str] = None) -> List[Prompt]:
        """Get prompts filtered by category and optionally subcategory."""
        if isinstance(category, dict):
            category = category["value"]
        if isinstance(subcategory, dict):
            subcategory = subcategory["value"]
            
        prompts = self.prompts.get(category, [])
        if subcategory:
            prompts = [p for p in prompts if p.subcategory == subcategory]
        return prompts

    def get_prompt_choices(self, category: str, subcategory: Optional[str] = None) -> List[str]:
        """Get list of prompt texts for dropdown display."""
        prompts = self.get_prompts(category, subcategory)
        return [p.text for p in prompts]

    def get_negative_prompt(self, category: str, subcategory: str, prompt_text: str) -> str:
        """Get matching negative prompt for the given prompt text."""
        if isinstance(category, dict):
            category = category["value"]
        if isinstance(subcategory, dict):
            subcategory = subcategory["value"]
            
        prompts = self.get_prompts(category, subcategory)
        for p in prompts:
            if p.text == prompt_text:
                return p.negative
        return "music, unwanted sounds"  # default