# CLAUDE.md

Project-specific instructions for Claude. Read this before making changes.

## What this is

`adb262.github.io` — Allan Bishop's personal site. Static, served by GitHub
Pages from the `main` branch. Built with Jekyll via the `github-pages` gem
(GitHub Pages' classic "deploy from a branch" mode does the build server-side).

## Hard constraints

- **Ruby is pinned to 3.3** locally because `github-pages` (v232) requires
  Ruby >= 2.6 and < 4.0. Don't suggest upgrading to Ruby 4 unless we also drop
  the `github-pages` gem (which would mean switching to a custom GitHub Actions
  workflow). PATH is set in `~/.zshrc`.
- **No Jekyll theme.** `_config.yml` has `theme: null` to stop `github-pages`
  from defaulting to `jekyll-theme-primer`, which silently ships its own
  `assets/css/style.scss` and would clobber our static stylesheet. If you ever
  see CSS suddenly broken after a build, the theme is the first thing to check.
- **CSS path is `assets/css/site.css`,** not `style.css`, specifically to
  avoid path collisions with any theme that might re-appear.
- **`Gemfile.lock` is gitignored** so the GitHub Pages builder resolves its
  own dependencies. Don't commit it.
- **The home page (`index.html`) is hand-written.** No frontmatter, no layout,
  no Liquid. Jekyll passes it through verbatim. Don't refactor it onto a layout
  without explicit user approval.

## Local preview

```bash
bundle exec jekyll serve --livereload
# → http://localhost:4000
```

`_config.yml` changes don't hot-reload — kill and restart the server.

## Directory layout

```
.
├── index.html                  # home page (hand-written, no Jekyll layout)
├── 404.html                    # standalone 404, links to /assets/css/site.css
├── _config.yml                 # Jekyll config; restart server on changes
├── _layouts/
│   └── blog.html               # layout for blog posts
├── assets/
│   ├── css/site.css            # all site CSS (do NOT rename)
│   └── images/                 # square thumbnails for the writing list
├── blogs/                      # long-form posts, one folder per post
│   ├── README.md               # author guide (excluded from build)
│   └── <slug>/
│       ├── index.md            # frontmatter + markdown body
│       └── *.{png,jpg,svg}     # co-located assets
├── Gemfile                     # pinned to `github-pages ~> 232`
├── README.md                   # human-facing project README
└── CLAUDE.md                   # this file
```

## Adding a new blog post

1. **Pick a slug.** Use `snake_case` or `kebab-case`. The folder name becomes
   the URL: `blogs/<slug>/` → `/blogs/<slug>/`.

2. **Create the folder and `index.md`:**

   ```
   blogs/<slug>/
   ├── index.md
   └── figure-1.png            # any co-located assets
   ```

3. **Frontmatter contract:**

   ```yaml
   ---
   layout: blog                  # required — picks _layouts/blog.html
   title: "Title in Title Case"  # required — page <title> + h1
   date: 2026-03-20              # required — ISO date
   description: "One sentence."   # optional — meta description for SEO
   math: true                    # optional — loads MathJax 3 when present
   read_minutes: 10              # optional — override the auto-computed estimate
   ---
   ```

4. **Reference images with bare relative paths:** `![alt](figure-1.png)`.
   For captioned figures, drop into HTML — Kramdown leaves it alone:

   ```html
   <figure>
     <img src="figure-1.png" alt="Descriptive alt text.">
     <figcaption>Caption text.</figcaption>
   </figure>
   ```

   Always write real `alt` text. Notion exports leave it blank — fix it.

5. **Add the writing-list entry to `index.html`** at the top of
   `<ol class="writing-list">` (entries are reverse-chronological):

   ```html
   <li class="writing-item">
       <a class="writing-thumb" href="/blogs/<slug>/">
           <img src="assets/images/<thumbnail>.svg" alt="" loading="lazy">
       </a>
       <div class="writing-body">
           <div class="writing-meta">Month D, YYYY</div>
           <h3 class="writing-title">
               <a href="/blogs/<slug>/">Title in Title Case</a>
           </h3>
           <p class="writing-desc">One- or two-sentence summary.</p>
       </div>
   </li>
   ```

   Drop a 1:1 thumbnail into `assets/images/<thumbnail>.svg` (or `.png`).
   The thumbnail is independent of any image inside the post.

## Read-time estimates

`_layouts/blog.html` shows a `~N min read` indicator next to the date. The
computation is:

1. If `read_minutes:` is set in the frontmatter, use it verbatim.
2. Otherwise, `ceil(word_count / 200)` (200 wpm, rounded up, minimum 1 min).

Use the `read_minutes:` override when the auto-estimate is misleading:

- **Dense math or proofs.** Reading-and-grokking is much slower than reading.
- **Lots of code.** A 200-word post with a 100-line code listing is not a
  1-minute read.
- **Long footnotes / asides.** Word count includes them; effective reading
  time may not.

If you don't have a strong reason to override, let the auto-compute run.

## Math (LaTeX)

Set `math: true` in the frontmatter. Then:

- Inline: `$E = mc^2$` or `\( E = mc^2 \)`
- Display: `$$ \int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2} $$`

MathJax 3 only loads on pages where `math: true` — non-math posts stay light.

## CSS conventions

- All site CSS lives in `assets/css/site.css`. One file, ordered top-down
  (tokens → layout → components → blog/`.prose` → responsive).
- Reuse design tokens from `:root` (`--bg`, `--fg`, `--muted`, `--rule`,
  `--accent`, `--serif`, `--sans`, `--max-width`). Don't introduce one-off
  colors or fonts without strong justification.
- Blog-post typography is scoped under `.prose` so it can't bleed into the
  home page or other layouts.
- Mobile breakpoint is `@media (max-width: 600px)` at the bottom of the file.
  When you add a desktop style, check whether it needs a mobile override.

## Things to avoid

- Don't add `Gemfile.lock` to git.
- Don't set `theme:` to anything except `null` without understanding the
  Primer-overwrites-our-CSS failure mode.
- Don't rename `assets/css/site.css` back to `style.css` (see "Hard constraints").
- Don't apply a Jekyll layout to `index.html` — it's intentionally raw HTML.
- Don't proactively commit. Wait for the user to ask. Surface what's
  uncommitted in the response when work is done.
