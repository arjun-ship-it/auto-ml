"""Tests for the AutoML Agent."""

import pytest
from unittest.mock import AsyncMock, patch

from app.agent.core import AutoMLAgent
from app.execution.safety import validate_code


class TestCodeSafety:
    """Test code safety checker."""

    def test_safe_code_passes(self):
        code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3]})
result = df.describe()
"""
        result = validate_code(code)
        assert result["safe"] is True

    def test_os_import_blocked(self):
        code = "import os\nos.system('rm -rf /')"
        result = validate_code(code)
        assert result["safe"] is False
        assert any("os" in issue for issue in result["issues"])

    def test_subprocess_blocked(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        result = validate_code(code)
        assert result["safe"] is False

    def test_network_access_blocked(self):
        code = "import requests\nrequests.get('http://evil.com')"
        result = validate_code(code)
        assert result["safe"] is False

    def test_eval_blocked(self):
        code = "eval('__import__(\"os\").system(\"whoami\")')"
        result = validate_code(code)
        assert result["safe"] is False


class TestAgent:
    """Test AutoML Agent initialization."""

    def test_agent_creation(self):
        agent = AutoMLAgent("test-project")
        assert agent.project_id == "test-project"

    @pytest.mark.asyncio
    async def test_agent_initialize(self):
        agent = AutoMLAgent("test-project")
        with patch.object(agent.conversation, 'load_history', new_callable=AsyncMock):
            with patch.object(agent.memory, 'load', new_callable=AsyncMock):
                await agent.initialize()
