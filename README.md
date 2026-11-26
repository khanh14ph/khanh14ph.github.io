# khanh14ph.github.io

Personal blog, built with [Jekyll](https://jekyllrb.com/) and hosted on GitHub Pages.

The design is a small hand-written theme built around the idea of a worklog:

- A left gutter holds dates, separated from the text by a thin margin line, like a logbook.
- Prose stays in one narrow column. Code blocks and wide figures may run out into the right margin, which is also where sidenotes go.
- Type is [IBM Plex](https://github.com/IBM/plex): Serif for titles, Sans for text, Mono for code.
- Colours are a cool white page, green links and a little copper, with a dark version that follows the reader's system setting.
- The mark in the header (and the favicon) is a 4x4 block with one row and one column picked out: a matrix product.

There is no JavaScript apart from MathJax and Mermaid, which load only on pages that use them.

## Layout

| Path | What it holds |
| --- | --- |
| `_posts/` | Blog posts, `YYYY-MM-DD-title.md` |
| `_pages/` | `home.md` (`/`, intro text above the post list), `about.md` (`/about/`), `404.md` |
| `_layouts/` | `default` (page shell), `home` (post list), `post`, `page` |
| `_includes/` | `head.html`, `scripts.html` (MathJax/Mermaid), `mark.html` (the logo), `sidenote.html` |
| `assets/css/style.css` | The whole stylesheet; colours, fonts and track widths are variables near the top |
| `assets/fonts/plex/` | IBM Plex web fonts (OFL), Latin and Vietnamese subsets |
| `images/` | Post images and favicons |

## Writing a post

Create `_posts/YYYY-MM-DD-my-title.md`:

```yaml
---
title: "My Title"
date: 2026-08-04
categories:
  - GPU
---
```

- The first paragraph is the excerpt shown on the homepage.
- `categories` only affects the URL (`/gpu/my-title/`); it can be left out.
- Reference images with an absolute path, e.g. `![caption](/images/figure.png)`.
- Math: `$$ ... $$`, inline or on its own line (MathJax).
- Diagrams: a fenced code block with the language `mermaid`.
- Sidenotes go in the right margin on wide screens (elsewhere, tap the number to open them). The `id` must be unique within the post:

  ```liquid
  Some text.{% include sidenote.html id="1" text="A note with *markdown* in it." %}
  ```

## Running locally

```bash
docker compose up
```

Serves at http://localhost:4000 with live reload. Without Docker: `bundle install && bundle exec jekyll serve -l -H localhost`.
