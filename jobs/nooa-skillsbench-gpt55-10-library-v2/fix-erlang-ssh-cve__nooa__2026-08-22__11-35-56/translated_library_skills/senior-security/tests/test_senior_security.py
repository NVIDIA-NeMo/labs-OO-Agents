from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from senior_security import SeniorSecurity


def test_skill_instantiates_and_lists_resources():
    skill = SeniorSecurity()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert 'scripts/pentest_automator.py' not in skill._list_resources()
    assert 'run_pentest_automator' not in visible_doc
    assert hasattr(skill, 'pentest_automator')
    assert 'pentest_automator' in visible_doc
    assert 'scripts/security_auditor.py' not in skill._list_resources()
    assert 'run_security_auditor' not in visible_doc
    assert hasattr(skill, 'security_auditor')
    assert 'security_auditor' in visible_doc
    assert 'scripts/threat_modeler.py' not in skill._list_resources()
    assert 'run_threat_modeler' not in visible_doc
    assert hasattr(skill, 'threat_modeler')
    assert 'threat_modeler' in visible_doc


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.senior-security' in registry.loaded()
    finally:
        registry.close()
