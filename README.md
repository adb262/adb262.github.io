# adb262.github.io

Personal site &mdash; Allan Bishop. Static HTML/CSS, served by GitHub Pages from `main`.

## Structure

```
index.html                  bio (home page) + contact footer
writing/index.html          writing list + contact footer
timeline/index.html         CV-style timeline + contact footer
projects/index.html         projects list + contact footer
assets/css/site.css         academic theme
assets/images/              square thumbnails for the writing list (currently SVG placeholders)
```

Each top-level page shares the same nav (Bio / Writing / Timeline / Projects /
Contact) and contact footer. Contact is an in-page anchor (`#contact`) to the
footer on whichever page you're on.

## Local preview

The site is built with Jekyll via the `github-pages` gem (matches what GitHub
Pages runs server-side). One-time setup:

```bash
brew install ruby@3.3
echo 'export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="/opt/homebrew/lib/ruby/gems/3.3.0/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="$HOME/.local/share/gem/ruby/3.3.0/bin:$PATH"' >> ~/.zshrc
exec zsh
gem install --user-install bundler
bundle install
```

Then to preview locally:

```bash
bundle exec jekyll serve --livereload
```

Then open <http://localhost:4000>.

## Adding a new piece of writing

Add a new `<li class="writing-item">` to the `.writing-list` in `writing/index.html`. Items are listed reverse-chronological. Drop a square thumbnail in `assets/images/`.
