# Repository Hygiene

This repo is configured to keep the public GitHub surface source-first and reproducible.

Do not commit:

- Raw or processed datasets
- Research notebooks with rendered images or private paths
- Model checkpoints, weights, and exported models
- Training outputs, logs, experiment trackers, and benchmark artifacts
- `.env` files, credentials, tokens, private keys, or cloud config files

Before publishing:

```bash
git status --short --ignored
git ls-files | rg "(Solar Dataset|Notebooks|Models|checkpoints|\\.pt$|\\.pth$|\\.ipynb$|__pycache__)"
git grep -n -I -i -E "(api[_-]?key|secret|token|password|credential|private[_-]?key|BEGIN .*PRIVATE|github_pat|sk-[A-Za-z0-9]|AKIA|AIza)"
```

If a secret was ever committed, remove it from history and rotate it. A normal delete commit is not enough for credentials that have already been pushed.
