from pathlib import Path

from nooa.agentdoc import doc
from nooa.skill_registry import SkillRegistry
from video_processor import VideoProcessor


def test_skill_instantiates_and_lists_resources():
    skill = VideoProcessor()
    visible_doc = doc(skill)
    assert 'list_resources' not in visible_doc
    assert 'read_resource' not in visible_doc
    assert 'run_resource_script' not in visible_doc
    assert 'scripts/process_video.py' not in skill._list_resources()
    assert 'run_process_video' not in visible_doc
    assert hasattr(skill, 'get_video_duration')
    assert 'get_video_duration' in visible_doc
    assert hasattr(skill, 'load_segments')
    assert 'load_segments' in visible_doc
    assert hasattr(skill, 'calculate_keep_segments')
    assert 'calculate_keep_segments' in visible_doc
    assert hasattr(skill, 'build_ffmpeg_filter')
    assert 'build_ffmpeg_filter' in visible_doc
    assert hasattr(skill, 'process_video')
    assert 'process_video' in visible_doc


def test_skill_registry_loads_package():
    class Agent:
        pass

    package_dir = Path(__file__).resolve().parents[1]
    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_dir.parent)
        assert 'local.video-processor' in registry.loaded()
    finally:
        registry.close()
