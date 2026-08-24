from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from audio_extractor import AudioExtractor


def test_skill_instantiates_and_lists_resources():
    skill = AudioExtractor()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert 'scripts/extract_audio.py' not in skill._list_resources()
    assert 'run_extract_audio' not in visible_doc
    assert hasattr(skill, 'extract_audio')
    assert 'extract_audio' in visible_doc
    assert hasattr(skill, 'extract_audio_2')
    assert 'extract_audio_2' in visible_doc


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.audio-extractor' in registry.loaded()
    finally:
        registry.close()
