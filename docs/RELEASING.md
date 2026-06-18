# Releasing

This documents a **prototype** CI-published release flow, proposed as a replacement for the in-repo, cumulative P2 update site. It mirrors the approach proven in [ponder-lab/Common-Eclipse-Refactoring-Framework](https://github.com/ponder-lab/Common-Eclipse-Refactoring-Framework) (see ponder-lab/Hybridize-Functions-Refactoring#629).

## Current Model (Committed Update Site)

The P2 update site lives **inside the source repo** (`edu.cuny.hunter.hybridize.updatesite/`) and is **cumulative**: every release commits its feature/plugin jars and rewrites `artifacts.jar`/`content.jar`, served from `raw.githubusercontent.com/.../main/...updatesite`. It currently holds tens of MB of binaries (it also bundles dependency jars such as WALA and PyDev).

Trade-offs:

- ➖ Binaries accumulate in `main` history **forever**—every release grows every clone, unrecoverably.
- ➖ `raw.githubusercontent.com` is not a real artifact host (no CDN/SLA).
- ➕ Zero infra; everything is in one repo.

## Prototype Model (This Branch: CI→GitHub Pages Composite Repo)

The update site is **built in CI and published to the `gh-pages` branch**, never committed to `main`. Each release publishes its **own** small p2 repo under `releases/X.Y.Z/`, and a p2 **composite repository** at the root ties them all together—so one stable URL exposes every version without a monolithic index.

Stable update-site URL (composite):

```
https://ponder-lab.github.io/Hybridize-Functions-Refactoring/releases/
```

Pieces added on this branch:

- **`.github/workflows/release.yml`**—manual-dispatch pipeline: set-version→build/test→tag + GitHub Release (with the p2 repo zipped as an asset)→publish to `gh-pages` as a composite→next-dev bump.
- **`tools/p2-composite.sh`**—regenerates `compositeContent.xml`/`compositeArtifacts.xml` from the per-version child dirs.

Unlike Common, this repo already has a `category.xml`, so the Tycho `eclipse-repository` build already produces a non-empty p2 repo—no `category.xml` fix is needed.

Trade-offs:

- ➕ **No binaries in `main`**—source history stays lean. The published bits live on the orphan `gh-pages` branch, independent of source clones.
- ➕ One CLI build produces the repo; no hand-assembled index.
- ➕ Composite layout scales to any number of versions; each release is an isolated, immutable child.
- ➕ Released p2 repo is also attached to the GitHub Release as a zip.
- ➖ Relies on GitHub Pages + Actions (infra, perms).

## Migration Notes (Not Done on This Branch)

1. Enable GitHub Pages for the `gh-pages` branch.
2. Seed `gh-pages` with the existing committed update site as `releases/archive/` (one child holding all versions through the latest release) so the Pages URL exposes all history alongside the old URL during transition.
3. Update the README's update-site URL to the Pages composite URL.
4. Drop the committed `edu.cuny.hunter.hybridize.updatesite` binaries from `main` (history preserved on `gh-pages`).
