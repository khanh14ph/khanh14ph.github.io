# khanh14ph.github.io

Personal blog, built with [Jekyll](https://jekyllrb.com/) and hosted on GitHub Pages.

The design is a small hand-written Tufte-style theme (cream page, [ET Book](https://github.com/edwardtufte/et-book) serif, a text column with a wide right margin for sidenotes), modelled on [siboehm.com](https://siboehm.com/). There is no JavaScript apart from MathJax and Mermaid, which load only on pages that use them.

## Layout

| Path | What it holds |
| --- | --- |
| `_posts/` | Blog posts, `YYYY-MM-DD-title.md` |
| `_pages/` | `home.md` (`/`, intro text above the post list), `about.md` (`/about/`), `404.md` |
| `_layouts/` | `default` (page shell), `home` (post list), `post`, `page` |
| `_includes/` | `head.html`, `scripts.html` (MathJax/Mermaid), `sidenote.html` |
| `assets/css/style.css` | The whole stylesheet; colours, fonts and column width are variables at the top |
| `assets/fonts/et-book/` | ET Book web fonts (MIT) |
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
- Sidenotes go in the right margin (tap the number to open them on a phone). The `id` must be unique within the post:

  ```liquid
  Some text.{% include sidenote.html id="1" text="A note with *markdown* in it." %}
  ```

## Running locally

```bash
docker compose up
```

Serves at http://localhost:4000 with live reload. Without Docker: `bundle install && bundle exec jekyll serve -l -H localhost`.
