from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from energy_calculator import EnergyCalculator


def test_skill_instantiates_and_lists_resources():
    skill = EnergyCalculator()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert 'scripts/calc_energy.py' not in skill._list_resources()
    assert 'run_calc_energy' not in visible_doc
    assert hasattr(skill, 'calc_energy')
    assert 'calc_energy' in visible_doc
    assert hasattr(skill, 'calculate_energy')
    assert 'calculate_energy' in visible_doc


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.energy-calculator' in registry.loaded()
    finally:
        registry.close()
