# 06 — Research RAG Intelligence

Local retrieval-and-grounding prototype separating retrieval from generation.

## Pipeline
Documents → chunking → TF-IDF retrieval → evidence ranking → answer context.

## Research Questions
Chunk-size effects, reranking quality, citation coverage and unsupported-claim detection.

A production LLM can be connected later, but evidence retrieval remains independently evaluable.
