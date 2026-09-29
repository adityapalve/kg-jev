"""System One routing layer for knowledge graph question answering.

The controller (jev) makes typed, calibrated decisions; templates or an LLM write small SPARQL
queries against a schema slice; plain code executes them and carries bindings between hops.
"""

__version__ = "0.1.0"
