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
   `# until the PyPI release: pip install "tabagent @ git+..."` line from the README's Install section. Merge the
   change to `main` through a pull request, with CI green.
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

## Notes

- **The PyPI page is the README.** `[tool.hatch.metadata.hooks.fancy-pypi-readme]` in `pyproject.toml` turns its
  repository-relative links into `github.com/dolev31/TabAgent/blob/main/...` links and its images into
  `raw.githubusercontent.com/dolev31/TabAgent/main/...` URLs, so keep the files under `docs/assets/` where the
  README points.
- **PyPI versions are immutable.** If the upload job fails, fix the cause and use *Re-run failed jobs*. If a bad
  version was uploaded, yank it on PyPI and release the next patch version; a file name can never be reused.
- **A tag that does not match `__version__`** stops the workflow before anything is built. Delete it with
  `git push origin :refs/tags/v0.1.0` and `git tag -d v0.1.0`, fix the version, and tag again.
