# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media policy stays usable without the optional LLM and media stacks."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("statement", ["import nooa", "import nooa.runtime"])
def test_policy_import_without_llm_or_media_dependencies(statement):
    code = f"""
import importlib.abc
import sys
class MissingOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'litellm', 'openai', 'anthropic', 'PIL', 'numpy'}}:
            raise ImportError('optional dependency intentionally unavailable')
sys.meta_path.insert(0, MissingOptionalDependencies())
{statement}
from nooa.config.model_config import ModelConfig
from nooa.runtime.media_capture import _MediaBuffer
assert ModelConfig(media_inputs=['pdf']).media_inputs == ('pdf',)
assert 'media_inputs' in ModelConfig.model_json_schema()['properties']
assert _MediaBuffer(1, media_inputs=['image']).media_inputs == ('image',)
assert not any(m in sys.modules for m in ('litellm', 'nooa.unifiedllm'))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_policy_types_retain_identity_and_validation():
    from nooa import media_inputs as policy
    from nooa.config.model_config import ModelConfig
    from nooa.unifiedllm import media_inputs as public

    assert public.MediaInputs is policy.MediaInputs
    assert public.validate_media_inputs is policy.validate_media_inputs
    assert public.attachment_kind is policy.attachment_kind
    with pytest.raises(ValueError, match="duplicates"):
        ModelConfig(media_inputs=["image", "image"])
