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
- The workflow checks whether GitHub Pages is using `workflow` deployments before it builds
- To let the workflow switch GitHub Pages from branch-based `legacy` mode to `workflow` mode automatically, add a `PAGES_ADMIN_TOKEN` secret with repository admin access
- Without that secret, the workflow can detect a bad Pages mode but cannot change it; in that case set `Settings -> Pages -> Source` to `GitHub Actions` manually
- Do not use the branch-based Pages builder for this repo; it does not support `jekyll-scholar` and fails on `{% bibliography %}`
- If a Pages job logs `GitHub Pages: github-pages v232` and ends with `Unknown tag 'bibliography'`, that is the branch-based Pages builder, not this workflow build. Switch `Settings -> Pages -> Source` to `GitHub Actions` and rerun the workflow.

## Content TODOs

- Replace the placeholder teaser thumbnails in `assets/img/publication_preview/`
- Add a CV PDF if you want a downloadable resume link later
- Add blog posts or additional pages as new content appears
