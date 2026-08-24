from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from find_bugs import FindBugs


def test_skill_instantiates_and_lists_resources():
    skill = FindBugs()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert isinstance(skill._list_resources(), list)


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.find-bugs' in registry.loaded()
    finally:
        registry.close()
