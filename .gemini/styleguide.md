# Gemini Code Assist PR Review Styleguide
**Repository:** `RatioPath` (RationAI)
**Context:** This is a Python library for large-scale processing, analysis, and transformation of whole-slide pathology images (WSIs).
It is built on top of the Ray framework to enable distributed, fault-tolerant, and scalable pipelines across multi-node environments.

## 📝 General Comment Style
- Keep comments **short and actionable**.
- Prefer **bullet points** over long paragraphs.
- Point to specific lines or sections when possible.
- Suggest improvements, not rewrite entire snippets.
- Avoid repetition of what the code already clearly states.
- Defer to the repo’s existing conventions unless there’s a clear bug or inconsistency.

## 📚 Documentation Review Expectations
- Highlight missing or incomplete documentation whenever a PR introduces new public API surface.
- Treat new public modules, classes, functions, methods, CLI entry points, configuration options, and user-facing behaviors as documentation candidates.
- Flag missing Google-style docstrings for new public Python objects when they should be part of the library API.
- Flag missing user or reference documentation when a change adds new functionality that users need to discover or understand.
- Check that documentation examples use the real package import paths exposed by `ratiopath`.
- Mention when docstrings are too thin for rendered API docs and should include sections like `Args:`, `Returns:`, `Yields:`, `Raises:`, or `Examples:` when relevant.
- If a PR changes behavior, configuration, or usage without updating docs, leave a review comment asking for the missing documentation.

## ✅ Documentation Review Triggers
- New public function, class, or module added without docstrings.
- New feature added without corresponding user-facing docs or reference docs.
- New parameter, return behavior, exception behavior, or example-worthy workflow added without documentation updates.
- Renamed or relocated public API without documentation and examples being updated.
- Changes that affect MkDocs or generated API reference output without matching documentation changes.
