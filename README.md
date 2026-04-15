# Jiale Fu Personal Website

This branch contains a fresh personal website rebuild for `jialefu/jialefu.github.io` on top of the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme.

## Branch setup

- Rebuilt on a clean orphan branch: `al-folio-rebuild`
- Existing AcademicPages implementation was left untouched
- Theme base imported from `alshedivat/al-folio` at commit `af82cce72a2b6a46ba80ed89e8f13bf82b60890d`

## Local development

1. Install Ruby and Bundler.
2. Install dependencies:

```bash
bundle install
```

3. Start the site locally:

```bash
bundle exec jekyll serve
```

4. Open `http://127.0.0.1:4000`.

## Deployment

- GitHub Pages deployment is handled through `.github/workflows/deploy.yml`
- The workflow builds the Jekyll site and deploys `_site`
- The workflow is configured for `main`, `master`, and `al-folio-rebuild`

## Content TODOs

- Replace the placeholder teaser thumbnails in `assets/img/publication_preview/`
- Add a CV PDF if you want a downloadable resume link later
- Add more publications or project pages as new work appears
