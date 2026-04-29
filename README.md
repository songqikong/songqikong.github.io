# songqikong.github.io

> Personal website of **Songqi Kong** (孔松琦) — built with [Beautiful Jekyll](https://beautifuljekyll.com/) and a custom **OpenCode** dark terminal theme.

---

## Overview

A bilingual (EN/CN) personal site featuring:

- **Homepage** — Terminal-style profile card with education, internship, and research sections
- **Blog** — Technical writings on 3D vision, point cloud, and deep learning
- **Vibing** — Interactive coding toys built via Vibe Coding
- **Cycling** — Photo gallery from cycling trips

## Site Structure

```
├── _config.yml          # Jekyll config (theme, nav, comments, social)
├── _layouts/            # Page templates (homepage, post, page, minimal…)
├── _includes/           # Partial templates (header, footer, nav, comments…)
├── _posts/              # Blog posts (14 articles, 2023)
├── _data/               # Jekyll data files
├── assets/
│   ├── css/
│   │   └── opencode.css # OpenCode dark terminal theme
│   ├── js/
│   │   └── custom-animations.js  # Page transition & scroll animations
│   └── img/             # Images & thumbnails (144 PNG, 13 JPG)
├── index.html           # Homepage (terminal-style profile)
├── BLOG.md              # Blog listing page
├── VIBING.md            # Interactive toys gallery (5 items)
├── CYCLING.md           # Cycling photo gallery
├── vibing/
│   ├── 2048.md          # 2048 game
│   ├── flappybird.md    # Flappy Bird game
│   ├── germany-map.md   # Interactive Germany map
│   ├── japan-map.md     # Interactive Japan map
│   └── political-test.md # 8values political compass test (64 questions, 4 axes, 32 labels)
├── feed.xml             # RSS feed
└── 404.html             # Custom 404 page
```

## OpenCode Theme

The site uses a custom **OpenCode** dark terminal aesthetic (`assets/css/opencode.css`):

- Color palette: `#0D1117` (bg) → `#C9D1D9` (text) → `#58A6FF` (accent)
- Monospace typography (`JetBrains Mono`, `Menlo`, `Consolas`)
- Terminal-style UI elements: `$` prompts, `[*]` markers, status dots
- Fade-in animations with staggered delays
- Bilingual toggle (EN ↔ CN) with `localStorage` persistence

## Vibing Toys

Interactive projects in the [Vibing](/VIBING) gallery:

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

- **Static Site Generator**: [Jekyll](https://jekyllrb.com/) (GitHub Pages)
- **Theme**: [Beautiful Jekyll](https://beautifuljekyll.com/) + custom OpenCode dark theme
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

Content is personal. Site template based on [Beautiful Jekyll](https://beautifuljekyll.com/) (MIT License).
