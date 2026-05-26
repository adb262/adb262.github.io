# Blogs

Long-form writing lives here. Each post is its own folder so that markdown
and assets stay co-located.

## Authoring a new post

1. Create a folder under `blogs/` named with the post slug. Use kebab-case
   (`adaptive-llms-via-evolution`) or snake_case (`underspecified_llms`) —
   whichever reads better. The folder name becomes the URL:

   ```
   blogs/<slug>/index.md   →   https://adb262.github.io/blogs/<slug>/
   ```

2. Inside the folder, create `index.md` with this frontmatter contract:

   ```yaml
   ---
   layout: blog                  # required — picks _layouts/blog.html
   title: "Title in Title Case"  # required — page <title> + h1
   date: 2026-03-20              # required — ISO date, shown above the title
   description: "One sentence."   # optional — meta description for SEO
   math: true                    # optional — loads MathJax 3 when present
   ---
   ```

3. Drop any images/PDFs/data files in the same folder. Reference them with
   relative paths:

   ```markdown
   ![alt text](figure-1.png)
   ```

   For captioned figures, drop down to raw HTML — Kramdown will leave it alone:

   ```html
   <figure>
     <img src="figure-1.png" alt="...">
     <figcaption>Caption text.</figcaption>
   </figure>
   ```

4. Link the post from the writing list in `index.html`:

   ```html
   <a class="writing-thumb" href="/blogs/<slug>/">
     <img src="assets/images/<thumbnail>.svg" alt="" loading="lazy">
   </a>
   ```

   The square thumbnail in `assets/images/` is shown on the home page; the
   image inside the post folder is shown inline. They don't need to match.

## Math (LaTeX)

Set `math: true` in the frontmatter to enable MathJax 3. Then:

- Inline: `$E = mc^2$` or `\( E = mc^2 \)`
- Display: `$$ \int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2} $$`

MathJax only loads on pages where `math: true`, so non-math posts stay light.

## Code

Fenced code blocks render with the site's mono font and a subtle panel:

````
```python
def hello(name: str) -> str:
    return f"hi, {name}"
```
````

## Local preview

```bash
bundle exec jekyll serve --livereload
```

Visit <http://localhost:4000/blogs/<slug>/>.
