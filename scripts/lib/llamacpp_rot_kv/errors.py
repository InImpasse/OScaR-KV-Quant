"""Lightweight errors for llama.cpp rotation calibration."""

from __future__ import annotations


class MissingDependencyError(RuntimeError):
    pass


class ToolExecutionError(RuntimeError):
    pass
