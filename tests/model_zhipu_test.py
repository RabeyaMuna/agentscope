# -*- coding: utf-8 -*-
"""Unit tests for Zhipu AI model class."""
from typing import AsyncGenerator
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch

from pydantic import BaseModel

from agentscope.message import TextBlock, ToolUseBlock
from agentscope.model import ZhipuChatModel, ChatResponse


class SampleModel(BaseModel):
    """Sample Pydantic model for testing structured output."""

    name: str
    age: int


class TestZhipuChatModel(IsolatedAsyncioTestCase):
    """Test cases for ZhipuChatModel."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch("agentscope.models.zhipu_model.zai.ZhipuAiClient") as mock_client:
            model = ZhipuChatModel(model_name="glm-4", api_key="test_key")
            self.assertEqual(model.model_name, "glm-4")
            self.assertTrue(model.stream)
            self.assertEqual(model.generate_kwargs, {})
            mock_client.assert_called_once_with(
                api_key="test_key",
                base_url=None,
            )

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        generate_kwargs = {"temperature": 0.7, "max_tokens": 1000}
        with patch("agentscope.models.zhipu_model.zai.ZhipuAiClient") as mock_client:
            model = ZhipuChatModel(
                model_name="glm-4-plus",
                api_key="test_key",
                stream=False,
                base_url="https://custom.api.url",
                generate_kwargs=generate_kwargs,
                timeout=30,
            )
            self.assertEqual(model.model_name, "glm-4-plus")
            self.assertFalse(model.stream)
            self.assertEqual(model.generate_kwargs, generate_kwargs)
            mock_client.assert_called_once_with(
                api_key="test_key",
                base_url="https://custom.api.url",
                timeout=30,
            )

    async def test_call_with_regular_model(self) -> None:
        """Test calling a regular model."""
        with patch("agentscope.models.z
