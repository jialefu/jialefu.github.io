$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$rubyBin = "C:\Ruby33-x64\bin"
$imageMagickBin = "C:\Program Files\ImageMagick-7.1.2-Q16"

if (Test-Path $rubyBin) {
  $env:PATH = "$rubyBin;$env:PATH"
}

if (Test-Path $imageMagickBin) {
  $env:PATH = "$imageMagickBin;$env:PATH"
}

Set-Location $repoRoot

if (-not (Test-Path (Join-Path $repoRoot "vendor\bundle"))) {
  bundle config set --local path vendor/bundle
  bundle install
}

bundle exec jekyll serve --config _config.yml,_config_local.yml --host 127.0.0.1 --port 4000 --livereload
