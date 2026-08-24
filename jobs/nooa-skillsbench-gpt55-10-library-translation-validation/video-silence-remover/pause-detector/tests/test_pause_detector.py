from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from pause_detector import PauseDetector


def test_skill_instantiates_and_lists_resources():
    skill = PauseDetector()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert 'scripts/detect_pauses.py' not in skill._list_resources()
    assert 'run_detect_pauses' not in visible_doc
    assert hasattr(skill, 'detect_pauses')
    assert 'detect_pauses' in visible_doc
    assert hasattr(skill, 'detect_pauses_2')
    assert 'detect_pauses_2' in visible_doc


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.pause-detector' in registry.loaded()
    finally:
        registry.close()
