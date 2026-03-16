// Small initializer for Mermaid and KaTeX behavior used by mkdocs site.
// Place this file in docs/js/ and register it in `extra_javascript` if you
// need a custom configuration.

// Mermaid init
if (typeof mermaid !== 'undefined') {
  mermaid.initialize({ startOnLoad: true, theme: 'default' });
}

// KaTeX auto-render default options (auto-render script must be loaded)
if (typeof renderMathInElement !== 'undefined') {
  // Render math in the whole document when called manually
  const options = {
    // Delimiters set to support $...$ and $$...$$
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false}
    ],
    // Add any other KaTeX auto-render options here
  };
  document.addEventListener('DOMContentLoaded', function () {
    renderMathInElement(document.body, options);
  });
}
