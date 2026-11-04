"""_compat — bodies for the legacy flat module paths that need real re-export logic.

Most pre-2026-07 flat paths (`mabel.wbc`, `mabel.swerve`, `mabel.self_collision`, …)
are a 1:1 rename onto a single new module, so the lazy finder in `mabel/__init__.py`
aliases them directly — no file needed. The few that AGGREGATE symbols from several
new homes (e.g. the old `mabel.dynamics`, split across model / plant / regions /
compliance) keep a small re-export body here, and the finder maps the old name onto it.

Do not import from `mabel._compat` in new code — import from the canonical homes named
in each module's docstring.
"""
