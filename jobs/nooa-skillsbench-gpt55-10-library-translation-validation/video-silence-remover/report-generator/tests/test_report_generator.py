from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from report_generator import ReportGenerator


def test_skill_instantiates_and_lists_resources():
    skill = ReportGenerator()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert 'scripts/generate_report.py' not in skill._list_resources()
    assert 'run_generate_report' not in visible_doc
    assert hasattr(skill, 'generate_report')
    assert 'generate_report' in visible_doc
    assert hasattr(skill, 'get_duration')
    assert 'get_duration' in visible_doc
    assert hasattr(skill, 'generate_report_2')
    assert 'generate_report_2' in visible_doc


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.report-generator' in registry.loaded()
    finally:
        registry.close()
