# Releasing TabAgent

A version tag publishes the package: pushing `v0.1.0` runs `.github/workflows/release.yml`, which checks that the
tag matches the package version, builds the sdist and wheel with `uv build`, checks them with `twine check`,
uploads them to [PyPI](https://pypi.org/project/tabagent/) through Trusted Publishing (no API token is stored
anywhere) and creates the GitHub Release with the same files. The workflow does nothing while the repository is
private.

## One-time setup, before the first release

1. **Make the repository public.** The release workflow, like the documentation site, is skipped while it is
   private.
2. **Register the trusted publisher on PyPI.** Log in to [pypi.org](https://pypi.org) with the account that will
   own the project (two-factor authentication on), open
   [Your account → Publishing](https://pypi.org/manage/account/publishing/) and, under *Add a new pending
   publisher*, choose the **GitHub** tab and enter exactly:

    | Field | Value |
    |---|---|
    | PyPI Project Name | `tabagent` |
    | Owner | `dolev31` |
    | Repository name | `TabAgent` |
    | Workflow name | `release.yml` |
    | Environment name | `pypi` |

    A pending publisher does not reserve the name: the project is created, and owned by that account, at the
    first successful upload. Add the other maintainers under the project's *Collaborators* page afterwards.
3. **Create the `pypi` environment on GitHub** (recommended; the first run would otherwise create it without
   protection): *Settings → Environments → New environment* `pypi`. Under *Deployment branches and tags*, allow
   only tags matching `v*`; optionally add required reviewers, so each upload waits for an approval.

## Each release

1. **Bump the version** in `src/tabagent/__init__.py` (`__version__`, [PEP 440](https://peps.python.org/pep-0440/):
   `0.1.0`, `0.2.0rc1`, ...). For the first release, also remove the
   `# until the PyPI release: pip install "tabagent @ git+..."` line from the README's Install section. In
   `CHANGELOG.md`, move the *Unreleased* entries under a heading for the new version and add the release date.
   Merge the change to `main` through a pull request, with CI green.
2. **Check the build locally** (optional; the workflow repeats it):

    ```bash
    rm -rf dist && uv build && uvx twine check --strict dist/*
    ```

3. **Tag `main` and push the tag:**

    ```bash
    git switch main && git pull
    git tag -a v0.1.0 -m "TabAgent 0.1.0"
    git push origin v0.1.0
    ```

4. **Watch the `release` workflow** under *Actions*. When it is green, the version is on
   [pypi.org/project/tabagent](https://pypi.org/project/tabagent/) and under *Releases*, with notes generated
   from the merged pull requests (edit them there if needed). Tags with `a`, `b`, `rc` or `dev` are marked as
   pre-releases. Confirm with `pip install tabagent==0.1.0` in a fresh environment.

## The explainer video

The landing page plays a 73-second explainer video (muted, looping, from when it scrolls into view; paused for readers
who ask for reduced motion), and the README opens with an animated cut of it. The MP4 (about 6 MB), the WebM
and the captions (VTT) are not committed, because the repository caps files at 500 KB: they are assets of a GitHub
Release, which the `pages` workflow downloads into `docs/assets/video/` before it builds the site. It takes the
newest release tagged `media-v*` that has the MP4, else any release that has it. Without one, the landing page shows
the animated pipeline in the video's place.

**At launch**, once the repository is public and Pages is enabled:

1. **Render the video** if `media/explainer/out/` does not hold it yet: see `media/explainer/README.md` (about
   12 minutes).
2. **Create the release and rebuild the site:**

    ```bash
    gh release create media-v1 --latest=false --title "Explainer video" \
      --notes "The explainer video of the documentation site, with English captions. Source: media/explainer/." \
      media/explainer/out/tabagent-explainer.mp4 media/explainer/out/tabagent-explainer.webm \
      media/explainer/out/tabagent-explainer.vtt
    gh workflow run pages.yml --ref main
    ```

    `--latest=false` keeps the package's version release as the repository's latest release, and the tag does
    not match `v*`, so nothing is published to PyPI. When the `pages` run is green, its log lists the three files
    and the landing page plays the video.
3. **The README's hero.** GitHub never autoplays a video in a README, so the README opens with an animated cut
   of the explainer instead: `explainer-teaser-hd.webp` (1280x720, 12 fps, about 1 MB), also an asset of
   `media-v1`. The `pages` workflow copies it to `site/assets/`, and the README loads it from the site, so it
   never enters git. A 720 px copy is committed as `docs/assets/explainer-teaser.webp` and is what the site
   serves under that name when the release has none.

A new cut of the video replaces the assets of the same release
(`gh release upload media-v1 media/explainer/out/tabagent-explainer.{mp4,webm,vtt} --clobber`, and the teaser
rebuilt from the new frames), followed by another run of `pages.yml`.

## Moving the repository

The project's address is `github.com/dolev31/TabAgent`, and the site's is `dolev31.github.io/TabAgent`. If the
repository moves to another owner, one command from the repository root rewrites every reference. Set `NEW` to the
new `owner/name`, and `NEW_SITE` to its site address, where GitHub Pages writes the owner in lower case:

```bash
export OLD=dolev31/TabAgent NEW=IBM/TabAgent OLD_SITE=dolev31.github.io/TabAgent NEW_SITE=ibm.github.io/TabAgent
git grep -lzIF -e "$OLD" -e "$OLD_SITE" | xargs -0 perl -pi -e 's/\Q$ENV{OLD}\E/$ENV{NEW}/g; s/\Q$ENV{OLD_SITE}\E/$ENV{NEW_SITE}/g'
```

It changes these files (`git grep -lIF -e "$OLD" -e "$OLD_SITE"` lists them):

| File | What holds the address |
|---|---|
| `README.md` | the badges, the install-from-git and clone commands, the site and video links |
| `pyproject.toml` | the project URLs, and the absolute links and images of the README on PyPI |
| `mkdocs.yml` | `site_url` and `repo_url`, from which the site's canonical URLs, Scholar tags, JSON-LD and GitHub links follow |
| `CITATION.cff` | `repository-code` and `url` |
| `.github/ISSUE_TEMPLATE/config.yml` | the reproduction guide and the private vulnerability report |
| `docs/index.md`, `docs/guides/*.md`, `docs/claims-map.md`, `docs/llms.txt` | clone commands and links to the repository and the site |
| `notebooks/quickstart.ipynb` | the Colab badge and the install cell |
| `experiments/tabschema_tau2.py` | the pull requests that its pipeline versions cite |
| `media/explainer/config.json`, `media/explainer/claims.md` | `repo_url`, shown on the video's end card, poster and thumbnail |
| `docs/releasing.md` | this page, including the command's `OLD` values, ready for the next move |

What the command cannot change:

- **The address in pixels.** The video's end card and captions, its poster and the README's thumbnail show it.
  Render them again (`render.py explainer captions stills`, see `media/explainer/README.md`), commit the two WebP
  files under `docs/assets/`, upload the new MP4, WebM and VTT to the release with `--clobber`, and upload the MP4
  again for the README's player.
- **Settings outside the repository:** the PyPI trusted publisher (its *Owner* and *Repository name*, above), the
  repository's homepage and Pages settings, the Search Console property for the new site address, and the links to
  the repository on the Hugging Face model cards and Space. The Hugging Face namespace (`huggingface.co/dolev31`)
  is separate and does not move.
- **Old links:** GitHub redirects `github.com` URLs after a transfer or a rename, but not `github.io` ones: the old
  site stops answering.

## Notes

- **The PyPI page is the README.** `[tool.hatch.metadata.hooks.fancy-pypi-readme]` in `pyproject.toml` turns its
  repository-relative links into `github.com/dolev31/TabAgent/blob/main/...` links and its images into
  `raw.githubusercontent.com/dolev31/TabAgent/main/...` URLs, so keep the files under `docs/assets/` where the
  README points.
- **PyPI versions are immutable.** If the upload job fails, fix the cause and use *Re-run failed jobs*. If a bad
  version was uploaded, yank it on PyPI and release the next patch version; a file name can never be reused.
- **A tag that does not match `__version__`** stops the workflow before anything is built. Delete it with
  `git push origin :refs/tags/v0.1.0` and `git tag -d v0.1.0`, fix the version, and tag again.
