"""Tests for a2a_langchain_adapters.toolkit."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a_langchain_adapters.toolkit import A2AToolkit

# ============================================================================
# A2AToolkit factory methods
# ============================================================================


class TestA2AToolkitFactory:
    """Tests for A2AToolkit.from_agent_urls factory method."""

    @pytest.mark.asyncio
    async def test_from_agent_urls_single_agent_success(self, agent_card):
        """Successfully load a single agent."""
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            mock_runnable = AsyncMock()
            mock_runnable.agent_card = agent_card
            MockRunnable.from_agent_url = AsyncMock(return_value=mock_runnable)

            toolkit = await A2AToolkit.from_agent_urls(["http://agent1:8080"])

            assert len(toolkit._runnables) == 1
            assert toolkit._runnables["Test Agent"] == mock_runnable
            MockRunnable.from_agent_url.assert_called_once_with(
                "http://agent1:8080", timeout=30.0, headers=None
            )

    @pytest.mark.asyncio
    async def test_from_agent_urls_multiple_agents_success(
        self, agent_card, agent_card_no_skills
    ):
        """Successfully load multiple agents."""
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            mock_runnable1 = AsyncMock()
            mock_runnable1.agent_card = agent_card
            mock_runnable2 = AsyncMock()
            mock_runnable2.agent_card = agent_card_no_skills

            MockRunnable.from_agent_url = AsyncMock(
                side_effect=[mock_runnable1, mock_runnable2]
            )

            toolkit = await A2AToolkit.from_agent_urls(
                [
                    "http://agent1:8080",
                    "http://agent2:8080",
                ]
            )

            assert len(toolkit._runnables) == 2
            assert toolkit._runnables["Test Agent"] == mock_runnable1
            assert toolkit._runnables["Simple Agent"] == mock_runnable2
            assert len(toolkit._agent_cards) == 2

    @pytest.mark.asyncio
    async def test_from_agent_urls_with_custom_timeout(self, agent_card):
        """Pass custom timeout to agent discovery."""
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            mock_runnable = AsyncMock()
            mock_runnable.agent_card = agent_card
            MockRunnable.from_agent_url = AsyncMock(return_value=mock_runnable)

            await A2AToolkit.from_agent_urls(
                ["http://agent1:8080"],
                timeout=60.0,
            )

            MockRunnable.from_agent_url.assert_called_once_with(
                "http://agent1:8080", timeout=60.0, headers=None
            )

    @pytest.mark.asyncio
    async def test_from_agent_urls_with_custom_headers(self, agent_card):
        """Pass custom headers to agent discovery."""
        headers = {"Authorization": "Bearer token123"}
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            mock_runnable = AsyncMock()
            mock_runnable.agent_card = agent_card
            MockRunnable.from_agent_url = AsyncMock(return_value=mock_runnable)

            await A2AToolkit.from_agent_urls(
                ["http://agent1:8080"],
                timeout=45.0,
                headers=headers,
            )

            MockRunnable.from_agent_url.assert_called_once_with(
                "http://agent1:8080", timeout=45.0, headers=headers
            )

    @pytest.mark.asyncio
    async def test_from_agent_urls_partial_failure(self, agent_card):
        """Handle partial failures gracefully."""
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            mock_runnable = AsyncMock()
            mock_runnable.agent_card = agent_card

            # First call succeeds, second fails
            MockRunnable.from_agent_url = AsyncMock(
                side_effect=[mock_runnable, Exception("Connection failed")]
            )

            toolkit = await A2AToolkit.from_agent_urls(
                [
                    "http://agent1:8080",
                    "http://agent2:8080",
                ]
            )

            # Only the successful agent should be loaded
            assert len(toolkit._runnables) == 1
            assert toolkit._runnables["Test Agent"] == mock_runnable

    @pytest.mark.asyncio
    async def test_from_agent_urls_all_failures(self):
        """Handle case where all agent discoveries fail."""
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            MockRunnable.from_agent_url = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            toolkit = await A2AToolkit.from_agent_urls(
                [
                    "http://agent1:8080",
                    "http://agent2:8080",
                ]
            )

            # Empty toolkit
            assert len(toolkit._runnables) == 0
            assert len(toolkit._agent_cards) == 0

    @pytest.mark.asyncio
    async def test_from_agent_urls_empty_list(self):
        """Handle empty URL list."""
        toolkit = await A2AToolkit.from_agent_urls([])

        assert len(toolkit._runnables) == 0
        assert len(toolkit._agent_cards) == 0


# ============================================================================
# Toolkit getters
# ============================================================================


class TestA2AToolkitGetters:
    """Tests for A2AToolkit getter methods."""

    def test_get_runnables(self, agent_card, agent_card_no_skills):
        """get_runnables returns all agent runnables."""
        toolkit = A2AToolkit()

        mock_runnable1 = MagicMock()
        mock_runnable1.agent_card = agent_card
        mock_runnable2 = MagicMock()
        mock_runnable2.agent_card = agent_card_no_skills

        toolkit._runnables["Test Agent"] = mock_runnable1
        toolkit._runnables["Simple Agent"] = mock_runnable2

        runnables = toolkit.get_runnables()

        assert len(runnables) == 2
        assert mock_runnable1 in runnables
        assert mock_runnable2 in runnables

    def test_get_runnables_empty(self):
        """get_runnables returns empty list when no agents."""
        toolkit = A2AToolkit()
        runnables = toolkit.get_runnables()

        assert runnables == []

    def test_get_runnable_by_name(self, agent_card):
        """get_runnable retrieves specific agent by name."""
        toolkit = A2AToolkit()
        mock_runnable = MagicMock()
        mock_runnable.agent_card = agent_card

        toolkit._runnables["Test Agent"] = mock_runnable

        result = toolkit.get_runnable("Test Agent")

        assert result == mock_runnable

    def test_get_runnable_by_name_not_found(self):
        """get_runnable returns None if agent not found."""
        toolkit = A2AToolkit()

        result = toolkit.get_runnable("Nonexistent Agent")

        assert result is None

    def test_get_agent_cards(self, agent_card, agent_card_no_skills):
        """get_agent_cards returns all discovered cards."""
        toolkit = A2AToolkit()

        toolkit._agent_cards["Test Agent"] = agent_card
        toolkit._agent_cards["Simple Agent"] = agent_card_no_skills

        cards = toolkit.get_agent_cards()

        assert len(cards) == 2
        assert agent_card in cards
        assert agent_card_no_skills in cards

    def test_get_agent_cards_empty(self):
        """get_agent_cards returns empty list when no agents."""
        toolkit = A2AToolkit()
        cards = toolkit.get_agent_cards()

        assert cards == []


# ============================================================================
# Tool derivation
# ============================================================================


class TestA2AToolkitTools:
    """Tests for A2AToolkit.get_tools (tool derivation)."""

    def test_get_tools_multiple_agents_with_skills(self, agent_card):
        """get_tools returns all agent tools."""
        toolkit = A2AToolkit()

        # Create mock runnables with tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "skill1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "skill2"

        mock_runnable1 = MagicMock()
        mock_runnable1.agent_card = agent_card
        mock_runnable1.as_tools = MagicMock(return_value=[mock_tool1, mock_tool2])

        mock_runnable2 = MagicMock()
        mock_runnable2.agent_card = agent_card
        mock_runnable2.as_tools = MagicMock(return_value=[mock_tool1])

        toolkit._runnables["Test Agent 1"] = mock_runnable1
        toolkit._runnables["Test Agent 2"] = mock_runnable2

        tools = toolkit.get_tools()

        assert len(tools) == 3  # 2 from first agent, 1 from second
        mock_runnable1.as_tools.assert_called_once()
        mock_runnable2.as_tools.assert_called_once()

    def test_get_tools_empty_toolkit(self):
        """get_tools returns empty list for empty toolkit."""
        toolkit = A2AToolkit()
        tools = toolkit.get_tools()

        assert tools == []

    def test_get_tools_single_agent(self, agent_card):
        """get_tools handles single agent."""
        toolkit = A2AToolkit()

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        mock_runnable = MagicMock()
        mock_runnable.agent_card = agent_card
        mock_runnable.as_tools = MagicMock(return_value=[mock_tool])

        toolkit._runnables["Test Agent"] = mock_runnable

        tools = toolkit.get_tools()

        assert len(tools) == 1
        assert tools[0] == mock_tool


# ============================================================================
# Lifecycle management
# ============================================================================


class TestA2AToolkitLifecycle:
    """Tests for A2AToolkit resource management."""

    @pytest.mark.asyncio
    async def test_close_all_runnables(self):
        """close() closes all underlying client wrappers."""
        toolkit = A2AToolkit()

        mock_runnable1 = MagicMock()
        mock_runnable1.close = AsyncMock()
        mock_runnable2 = MagicMock()
        mock_runnable2.close = AsyncMock()

        toolkit._runnables["Agent1"] = mock_runnable1
        toolkit._runnables["Agent2"] = mock_runnable2

        await toolkit.close()

        mock_runnable1.close.assert_called_once()
        mock_runnable2.close.assert_called_once()
        assert len(toolkit._runnables) == 0

    @pytest.mark.asyncio
    async def test_close_empty_toolkit(self):
        """close() handles empty toolkit gracefully."""
        toolkit = A2AToolkit()
        await toolkit.close()  # Should not raise

        assert len(toolkit._runnables) == 0

    @pytest.mark.asyncio
    async def test_close_one_fails(self):
        """close() continues if one runnable fails to close."""
        toolkit = A2AToolkit()

        mock_runnable1 = MagicMock()
        mock_runnable1.close = AsyncMock(side_effect=Exception("Close failed"))
        mock_runnable2 = MagicMock()
        mock_runnable2.close = AsyncMock()

        toolkit._runnables["Agent1"] = mock_runnable1
        toolkit._runnables["Agent2"] = mock_runnable2

        # Should raise the exception from first runnable
        with pytest.raises(Exception, match="Close failed"):
            await toolkit.close()

    @pytest.mark.asyncio
    async def test_context_manager_enter(self):
        """__aenter__ returns self."""
        toolkit = A2AToolkit()
        async with toolkit as ctx:
            assert ctx is toolkit

    @pytest.mark.asyncio
    async def test_context_manager_exit(self):
        """__aexit__ closes all resources."""
        toolkit = A2AToolkit()

        mock_runnable = MagicMock()
        mock_runnable.close = AsyncMock()
        toolkit._runnables["Test Agent"] = mock_runnable

        async with toolkit:
            pass

        mock_runnable.close.assert_called_once()


# ============================================================================
# Integration scenarios
# ============================================================================


class TestA2AToolkitIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_toolkit_workflow(self, agent_card):
        """Complete workflow: load, retrieve, and use agents."""
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            # Setup
            mock_runnable = AsyncMock()
            mock_runnable.agent_card = agent_card
            mock_tool = MagicMock()
            mock_tool.name = "test_tool"
            mock_runnable.as_tools = MagicMock(return_value=[mock_tool])

            MockRunnable.from_agent_url = AsyncMock(return_value=mock_runnable)

            # Load agents
            toolkit = await A2AToolkit.from_agent_urls(
                [
                    "http://agent1:8080",
                    "http://agent2:8080",
                ]
            )

            # Query runnables
            runnables = toolkit.get_runnables()
            assert len(runnables) >= 1

            # Get specific agent
            agent = toolkit.get_runnable("Test Agent")
            assert agent is not None

            # Derive tools
            tools = toolkit.get_tools()
            assert len(tools) >= 1

            # Get cards
            cards = toolkit.get_agent_cards()
            assert len(cards) >= 1

            # Cleanup
            await toolkit.close()

    @pytest.mark.asyncio
    async def test_toolkit_duplicate_agent_names(self, agent_card):
        """Handle case with duplicate agent names (last wins)."""
        with patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable:
            # Create two agents with the same name
            mock_runnable1 = MagicMock()
            mock_runnable1.agent_card = agent_card

            # Same name but different instance
            mock_runnable2 = MagicMock()
            mock_runnable2.agent_card = agent_card

            MockRunnable.from_agent_url = AsyncMock(
                side_effect=[mock_runnable1, mock_runnable2]
            )

            toolkit = await A2AToolkit.from_agent_urls(
                [
                    "http://agent1:8080",
                    "http://agent1:8080",  # Same name
                ]
            )

            # Last one wins
            assert toolkit.get_runnable("Test Agent") == mock_runnable2
            assert len(toolkit._runnables) == 1

    @pytest.mark.asyncio
    async def test_toolkit_logging_on_load(self, agent_card):
        """Verify logging on successful agent load."""
        with (
            patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable,
            patch("a2a_langchain_adapters.toolkit.logger") as mock_logger,
        ):
            mock_runnable = MagicMock()
            mock_runnable.agent_card = agent_card
            MockRunnable.from_agent_url = AsyncMock(return_value=mock_runnable)

            await A2AToolkit.from_agent_urls(["http://agent1:8080"])

            # Verify info log was called
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_toolkit_logging_on_failure(self):
        """Verify logging on agent load failure."""
        with (
            patch("a2a_langchain_adapters.toolkit.A2ARunnable") as MockRunnable,
            patch("a2a_langchain_adapters.toolkit.logger") as mock_logger,
        ):
            MockRunnable.from_agent_url = AsyncMock(
                side_effect=Exception("Network error")
            )

            await A2AToolkit.from_agent_urls(["http://agent1:8080"])

            # Verify exception log was called
            mock_logger.exception.assert_called()


# ============================================================================
# Initialization and state
# ============================================================================


class TestA2AToolkitState:
    """Tests for A2AToolkit state management."""

    def test_init_creates_empty_toolkit(self):
        """__init__ creates empty toolkit."""
        toolkit = A2AToolkit()

        assert toolkit._runnables == {}
        assert toolkit._agent_cards == {}

    def test_toolkit_state_isolation(self):
        """Multiple toolkit instances don't share state."""
        toolkit1 = A2AToolkit()
        toolkit2 = A2AToolkit()

        mock_runnable = MagicMock()
        mock_card = MagicMock()

        toolkit1._runnables["agent"] = mock_runnable
        toolkit1._agent_cards["agent"] = mock_card

        assert len(toolkit2._runnables) == 0
        assert len(toolkit2._agent_cards) == 0
