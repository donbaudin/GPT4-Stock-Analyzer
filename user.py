from typing import Optional, Dict
import json
import os

class UserSession:
    """
    Stores user profile, preferences, and session state for the GPT-4o Stock Analyzer Hybrid.
    """
    def __init__(
        self,
        profile: str = "Beginner",
        goal: str = "growth",
        risk: str = "medium",
        tone: str = "professional",
        accessibility: bool = False,
        mode_type: str = "guided",
        preferences: Optional[Dict] = None
    ):
        self.profile = profile
        self.goal = goal
        self.risk = risk
        self.tone = tone
        self.accessibility = accessibility
        self.mode_type = mode_type
        self.preferences = preferences or {}

    def save(self, path: str = "user_session.json") -> None:
        """
        Save the user session to a JSON file.
        """
        data = self.__dict__
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str = "user_session.json") -> Optional['UserSession']:
        """
        Load a user session from a JSON file.
        """
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return UserSession(**data)
