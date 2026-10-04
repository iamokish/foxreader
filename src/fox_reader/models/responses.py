from typing import List


class TranslationResult:
    """Base translation result used by all translation backends."""

    def __init__(
        self,
        code: int,
        id: int,
        data: str,
        source_lang: str,
        target_lang: str,
        method: str,
        message: str = "",
        alternatives: List[str] = None,
    ):
        self.code = code
        self.id = id
        self.message = message
        self.data = data
        self.alternatives = alternatives if alternatives is not None else []
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.method = method

    def to_dict(self) -> dict:
        res = {
            "code": self.code,
            "id": self.id,
            "data": self.data,
            "alternatives": self.alternatives,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "method": self.method,
        }
        if self.message:
            res["message"] = self.message
        return res
