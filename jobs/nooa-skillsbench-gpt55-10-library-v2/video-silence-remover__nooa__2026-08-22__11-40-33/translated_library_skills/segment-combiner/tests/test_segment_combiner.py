from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from segment_combiner import SegmentCombiner


def test_skill_instantiates_and_lists_resources():
    skill = SegmentCombiner()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert 'scripts/combine_segments.py' not in skill._list_resources()
    assert 'run_combine_segments' not in visible_doc
    assert hasattr(skill, 'combine_segments')
    assert 'combine_segments' in visible_doc


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.segment-combiner' in registry.loaded()
    finally:
        registry.close()
