"""Tests for FilePart support and file upload/download."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from a2a_langchain_adapters import A2AResult, A2ARunnable
from a2a_langchain_adapters.client_wrapper import A2AClientWrapper


class TestFileUpload:
    """Test file upload functionality."""

    @pytest.mark.asyncio
    async def test_send_file_with_text(self, client_wrapper):
        """Send file together with text message."""
        file_content = b"PDF content here"
        files = [("document.pdf", file_content, "application/pdf")]

        expected_result = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            text="Processed",
        )

        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await client_wrapper.send_message("Analyze this file", files=files)

        assert result.status == "completed"
        assert result.text == "Processed"

    @pytest.mark.asyncio
    async def test_send_multiple_files(self, client_wrapper):
        """Send multiple files in one message."""
        files = [
            ("file1.txt", b"content1", "text/plain"),
            ("file2.pdf", b"content2", "application/pdf"),
        ]

        expected_result = A2AResult(task_id="t1", context_id="c1", status="completed")

        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await client_wrapper.send_message("Process these files", files=files)

        assert result.status == "completed"


class TestFileDownload:
    """Test file download functionality."""

    def test_decode_file_bytes(self):
        """Decode base64-encoded file bytes from result."""
        file_content = b"PDF data"
        file_bytes_b64 = base64.b64encode(file_content).decode()

        file_data = {
            "name": "output.pdf",
            "mime_type": "application/pdf",
            "bytes": file_bytes_b64,
        }

        wrapper = A2AClientWrapper("http://test:8080")
        decoded = wrapper.decode_file_bytes(file_data)

        assert decoded == file_content

    def test_decode_file_bytes_invalid(self):
        """Raise error if file data has no 'bytes' field."""
        file_data = {
            "name": "output.pdf",
            "mime_type": "application/pdf",
            "uri": "https://example.com/output.pdf",
        }

        wrapper = A2AClientWrapper("http://test:8080")

        with pytest.raises(ValueError, match="no 'bytes' field"):
            wrapper.decode_file_bytes(file_data)

    @pytest.mark.asyncio
    async def test_download_file_from_uri(self):
        """Download file from URI."""
        file_content = b"downloaded content"

        with patch("a2a_langchain_adapters.client_wrapper.httpx") as mock_httpx:
            mock_response = AsyncMock()
            mock_response.content = file_content
            mock_response.raise_for_status = Mock()

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=mock_response)

            mock_httpx.AsyncClient.return_value = mock_client

            wrapper = A2AClientWrapper("http://test:8080")
            result = await wrapper.download_file("https://example.com/file.pdf")

            assert result == file_content
            mock_client.get.assert_called_once_with("https://example.com/file.pdf")


class TestFileResponse:
    """Test handling of file responses."""

    def test_a2a_result_with_files(self):
        """A2AResult can store file information."""
        result = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            text="Generated report",
            files=[
                {
                    "name": "report.pdf",
                    "mime_type": "application/pdf",
                    "uri": "https://example.com/report.pdf",
                }
            ],
        )

        assert len(result.files) == 1
        assert result.files[0]["name"] == "report.pdf"
        assert "uri" in result.files[0]

    def test_a2a_result_with_file_bytes(self):
        """A2AResult can store file bytes."""
        file_bytes_b64 = base64.b64encode(b"content").decode()

        result = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            files=[
                {
                    "name": "data.bin",
                    "mime_type": "application/octet-stream",
                    "bytes": file_bytes_b64,
                }
            ],
        )

        assert "bytes" in result.files[0]
        assert result.files[0]["bytes"] == file_bytes_b64


class TestToolWithFiles:
    """Test A2ARunnable.as_tool() with file responses."""

    @pytest.mark.asyncio
    async def test_tool_returns_file_info_json(self):
        """Tool returns JSON with file info if files present."""
        result = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            text="Generated report",
            files=[
                {
                    "name": "report.pdf",
                    "mime_type": "application/pdf",
                    "uri": "https://example.com/report.pdf",
                }
            ],
        )

        # Create a mock wrapper with an ainvoke that returns our result
        mock_wrapper = AsyncMock(spec=A2AClientWrapper)
        mock_wrapper.agent_card = None

        runnable = A2ARunnable(mock_wrapper)

        # Mock ainvoke to return our result
        with patch.object(runnable, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = result

            tool = runnable.as_tool()
            tool_result = await tool._arun("Generate report")

            # Result should be JSON with text + file info
            parsed = json.loads(tool_result)
            assert parsed["text"] == "Generated report"
            assert len(parsed["files"]) == 1
            assert parsed["files"][0]["name"] == "report.pdf"
            assert parsed["files"][0]["has_uri"] is True
            assert parsed["files"][0]["has_bytes"] is False

    @pytest.mark.asyncio
    async def test_tool_with_multiple_files(self):
        """Tool handles multiple file responses."""
        result = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            text="Generated files",
            files=[
                {
                    "name": "report.pdf",
                    "mime_type": "application/pdf",
                    "uri": "https://example.com/report.pdf",
                },
                {
                    "name": "data.csv",
                    "mime_type": "text/csv",
                    "uri": "https://example.com/data.csv",
                },
            ],
        )

        # Create a mock wrapper
        mock_wrapper = AsyncMock(spec=A2AClientWrapper)
        mock_wrapper.agent_card = None

        runnable = A2ARunnable(mock_wrapper)

        # Mock ainvoke to return our result
        with patch.object(runnable, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = result

            tool = runnable.as_tool()
            tool_result = await tool._arun("Generate files")

            parsed = json.loads(tool_result)
            assert len(parsed["files"]) == 2
            assert parsed["files"][0]["name"] == "report.pdf"
            assert parsed["files"][1]["name"] == "data.csv"

    @pytest.mark.asyncio
    async def test_tool_without_files(self):
        """Tool returns text normally if no files present."""
        result = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            text="No files generated",
            files=[],
        )

        # Create a mock wrapper
        mock_wrapper = AsyncMock(spec=A2AClientWrapper)
        mock_wrapper.agent_card = None

        runnable = A2ARunnable(mock_wrapper)

        # Mock ainvoke to return our result
        with patch.object(runnable, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = result

            tool = runnable.as_tool()
            tool_result = await tool._arun("Do something")

            # Should return plain text, not JSON
            assert tool_result == "No files generated"
            # Should not be JSON
            with pytest.raises(json.JSONDecodeError):
                json.loads(tool_result)


class TestRunableWithFiles:
    """Test A2ARunnable file upload integration."""

    @pytest.mark.asyncio
    async def test_ainvoke_with_files(self):
        """ainvoke() passes files to client."""
        file_content = b"test file"
        files = [("test.txt", file_content, "text/plain")]

        expected_result = A2AResult(
            task_id="t1",
            context_id="c1",
            status="completed",
            text="Processed",
        )

        mock_wrapper = AsyncMock(spec=A2AClientWrapper)
        mock_wrapper.send_message = AsyncMock(return_value=expected_result)

        runnable = A2ARunnable(mock_wrapper)

        result = await runnable.ainvoke("Process this", files=files)

        assert result.status == "completed"

        # Verify files were passed through to send_message
        mock_wrapper.send_message.assert_called_once()
        call_kwargs = mock_wrapper.send_message.call_args.kwargs
        assert call_kwargs["files"] == files
