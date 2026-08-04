# khanh14ph.github.io

Personal site, built with [Jekyll](https://jekyllrb.com/) and hosted on GitHub Pages.
Originally derived from [academicpages](https://github.com/academicpages/academicpages.github.io) (MIT), since trimmed to only what this site uses.

## Layout

| Path | What it holds |
| --- | --- |
| `_pages/` | Standalone pages: about (`/`), blog index, per-category blog pages, projects, books, research, 404 |
| `_posts/` | Blog posts, `YYYY-MM-DD-title.md` |
| `_data/` | `navigation.yml` (header links), `projects.yml`, `books.yml`, `ui-text.yml` |
| `_layouts/` | `default`, `single` (pages/posts), `archive` (listings), `compress` |
| `_includes/` | Shared fragments — masthead, author sidebar, post listing, SEO |
| `_sass/`, `assets/` | Styles and compiled JS; theme is set by `site_theme` in `_config.yml` |
| `images/` | Avatar and favicons |

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

The `categories` value routes the post to a section page. Existing sections and their
categories: GPU (`GPU`), Computer Architecture (`Computer Architecture`), OS
(`Operating System`), C++ (`C++`), Python (`Python`), Research (`Research`).

MathJax and Mermaid are available in posts (see `_includes/footer/custom.html`).

## Adding a section

1. Create `_pages/blog-<name>.html`:

   ```
   ---
   layout: archive
   permalink: /blog/<name>/
   title: "Blog: <Name>"
   category: <Category>
   ---

   {% include posts-by-category.html %}
   ```

2. Add the link to `_data/navigation.yml`.

## Running locally

```bash
docker compose up
```

Serves at http://localhost:4000 with live reload. Without Docker: `bundle install && bundle exec jekyll serve -l -H localhost`.
