# Jiale Fu Personal Website

This branch contains the rebuilt `jialefu/jialefu.github.io` website on top of the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme. The original AcademicPages site was left untouched on `master`; the new site lives on `al-folio-rebuild`.

## Structure

- Home page content lives in `_pages/about.md`
- Publications are maintained in `_bibliography/papers.bib`
- News items are maintained in `_news/`
- Social/contact links are configured in `_data/socials.yml`

## Local development

1. Install Ruby and Bundler.
2. Install dependencies:

```bash
bundle install
```

3. Start the local preview:

```powershell
.\preview.ps1
```

Or run Jekyll directly:

```bash
bundle exec jekyll serve --config _config.yml,_config_local.yml
```

4. Open `http://127.0.0.1:4000`.

## Deployment

- GitHub Pages deployment is handled through `.github/workflows/deploy.yml`
- The workflow builds the Jekyll site and deploys `_site` as a GitHub Pages artifact
- The workflow is configured for `main`, `master`, and `al-folio-rebuild`
- The workflow also checks GitHub Pages mode and will try to switch the repo from branch-based `legacy` builds to `workflow`
- If GitHub still refuses that automatic switch, set `Settings -> Pages -> Source` to `GitHub Actions`
- Do not use the branch-based Pages builder for this repo; it does not support `jekyll-scholar` and fails on `{% bibliography %}`

## Content TODOs

- Replace the placeholder teaser thumbnails in `assets/img/publication_preview/`
- Add a CV PDF if you want a downloadable resume link later
- Add blog posts or additional pages as new content appears
