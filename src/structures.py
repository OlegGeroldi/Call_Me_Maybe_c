"""Pydantic models for the two input files and the output records."""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict


class ParameterDefinition(BaseModel):
    """Definition of a single function parameter."""

    model_config = ConfigDict(extra="allow")

    type: str


class ReturnDefinition(BaseModel):
    """Definition of a function return value."""

    model_config = ConfigDict(extra="allow")

    type: str


class FunctionDefinition(BaseModel):
    """Definition of one callable function."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition


class FunctionCallingTest(BaseModel):
    """One input prompt."""

    model_config = ConfigDict(extra="allow")

    prompt: str


class FunctionCallResult(BaseModel):
    """One record of the output file."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, Any]
