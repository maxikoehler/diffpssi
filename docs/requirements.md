# Documentation build requirements

Install the recommended packages for building the docs locally in a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-mermaid2-plugin pymdown-extensions
```

Run local preview:

```bash
mkdocs serve
```

Notes:
- `mkdocstrings[python]` loads the python handler used to extract docstrings from the code base.
- `pymdownx.arithmatex` provides LaTeX support in Markdown and requires KaTeX JS/CSS (already included in `mkdocs.yaml` via CDN links).
- `mkdocs-mermaid2-plugin` enables server-side rendering or client-side integration (the plugin handles most setups). If diagrams don't appear, include `docs/js/mermaid-init.js` and register it in `extra_javascript` (already included in our example config).
