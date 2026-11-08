# songqikong.github.io

> Personal website of **Songqi(Carson) Kong** (孔松琦) — Jekyll on the AcademicPages layout chain, with a custom **OpenCode** design layer.

---

## Overview

A bilingual EN/CN personal site featuring:

- **Homepage** — Serif résumé timeline with work experience, education and publications
- **Blog** — Technical writings on 3D vision, point cloud, and deep learning
- **Miscellaneous** — Cycling photos and notes from outside work

The standalone toy pages under `vibing/` stay in the repository and remain reachable by their direct URLs, but intentionally have no navigation entry.

## Site Structure

```
├── _config.yml          # Jekyll config (nav, comments, social)
├── _layouts/            # Page templates (acad-single, home, page, post…)
├── _includes/           # Partial templates (masthead, footer, head, scripts…)
├── _posts/              # Blog posts (14 articles, 2023)
├── _data/               # Jekyll data files, including navigation.yml
├── _sass/               # Base theme sources, compiled into assets/css/acad.css
├── assets/
│   ├── css/
│   │   └── opencode.css # Design layer: tokens, oc-* components, chrome bindings
│   ├── js/
│   │   └── custom-animations.js  # Page transition & scroll animations
│   └── img/             # Images & thumbnails
├── index.html           # Homepage, serif résumé timeline
├── BLOG.md              # Blog listing, homepage visual style
├── MISCELLANEOUS.md     # Cycling photos + notes, homepage visual style
├── vibing/              # Standalone toys, no navigation entry
│   ├── 2048.md          # 2048 game
│   ├── flappybird.md    # Flappy Bird game
│   ├── germany-map.md   # Interactive Germany map
│   ├── japan-map.md     # Interactive Japan map
│   └── political-test.md # 8values political compass test (64 questions, 4 axes, 32 labels)
├── feed.xml             # RSS feed
└── 404.html             # Custom 404 page
```

## Styling

Two stylesheets, one layout chain:

- `assets/css/acad.css` — base theme compiled from `_sass/`, serif-first typography
- `assets/css/opencode.css` — `--oc-*` colour tokens, terminal accents, dark theme, masthead and footer bindings

Colours live only in `:root` / `html.dark-theme`; the base theme reads them through `var(--oc-*)`. Bilingual text is switched with `data-lang="en"` / `data-lang="cn"`, persisted in `localStorage`.

## Standalone Toys

Kept in the repository, reachable only by direct URL:

| Toy | Description |
|-----|-------------|
| 2048 | Classic sliding puzzle game |
| Flappy Bird | Side-scrolling arcade game |
| Germany Map | Interactive SVG map of Germany |
| Japan Map | Interactive SVG map of Japan |
| Political Test | 8values-style compass with 32 Chinese internet identity labels |

## Blog Topics

14 posts covering:

- **3D Vision**: SnowflakeNet, SCODA, P2C, PointGPT, PonderV2, TICC
- **Deep Learning**: Masked Autoencoders, Domain Adaptation vs Transfer Learning
- **Algorithms**: Blossom Algorithm, Multi-Resource Interleaving

## Tech Stack

- **Static Site Generator**: Jekyll (GitHub Pages)
- **Theme**: AcademicPages layout chain + custom OpenCode design layer
- **Comments**: Gitalk (GitHub Issues-based)
- **Analytics**: Busuanzi (visitor/page view counts)
- **Hosting**: GitHub Pages

## Local Development

```bash
# Install dependencies
bundle install

# Run local server
bundle exec jekyll serve

# Open at http://localhost:4000
```

## License

Content is personal. Site template based on AcademicPages / Beautiful Jekyll (MIT License).
