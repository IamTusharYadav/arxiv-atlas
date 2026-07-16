import importlib.resources
import re
from dataclasses import dataclass

import yaml

from atlas_agents.bedrock import HAIKU, SONNET

_KNOWN_MODELS = {HAIKU, SONNET}
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    model: str
    system: str
    changelog: tuple[str, ...]

    @property
    def tag(self) -> str:
        return f"{self.id}@{self.version}"

    def render(self, **params: object) -> str:
        # Cards with placeholders (planner, check) must be rendered; static cards call this
        # with no params and get the system text back untouched, never touching str.format.
        return self.system.format(**params) if params else self.system


def _load() -> dict[str, Prompt]:
    prompts: dict[str, Prompt] = {}
    for card in importlib.resources.files(__name__).iterdir():
        if not card.name.endswith(".yaml"):
            continue
        data = yaml.safe_load(card.read_text(encoding="utf-8"))
        stem = card.name[: -len(".yaml")]
        prompt = Prompt(
            id=str(data["id"]),
            version=str(data["version"]),
            model=str(data["model"]),
            system=str(data["system"]),
            changelog=tuple(str(c) for c in data.get("changelog", [])),
        )
        if prompt.id != stem:
            raise ValueError(f"prompt card {card.name}: id {prompt.id!r} must match filename")
        if not _SEMVER.match(prompt.version):
            raise ValueError(f"prompt card {card.name}: version {prompt.version!r} is not semver")
        if prompt.model not in _KNOWN_MODELS:
            raise ValueError(f"prompt card {card.name}: unknown model {prompt.model!r}")
        prompts[prompt.id] = prompt
    return prompts


_PROMPTS = _load()


def get(prompt_id: str) -> Prompt:
    return _PROMPTS[prompt_id]


PLANNER = get("planner")
RERANKER = get("reranker")
EXTRACTOR = get("extractor")
SYNTHESIZER = get("synthesizer")
CHECK = get("check")
JUDGE = get("judge")
DIRECTION = get("direction")
LANDSCAPE = get("landscape")
