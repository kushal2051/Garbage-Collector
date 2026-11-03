# Shared skills

Keep canonical skill files here, grouped by category. This repository has one category,
`anomaly`; the convention and the sync script are carried over from
[KGB-CIA](https://github.com/UNC-RENCI-KGAI/KGB-CIA), where categories mirror the agent
packages.

Each skill is a directory containing `SKILL.md`, and keeps its own supporting references,
scripts and assets alongside it:

```text
.agents/skills/
└── anomaly/
    ├── kg-anomaly-pipeline/
    │   └── SKILL.md
    ├── read-eval-results/
    │   └── SKILL.md
    └── slurm-run/
        └── SKILL.md
```

Use a globally unique leaf directory name, with matching `name` frontmatter and a nonempty
`description`. Category names are **not** part of the skill name:

```markdown
---
name: kg-anomaly-pipeline
description: Run or extend the RotatE one-class-SVM garbage-collection pipeline.
---

Instructions go here. Reference supporting files relative to this directory.
```

To expose these to Claude Code as flat directory entries, create relative symlinks in
`.claude/skills/`:

```bash
python3 scripts/sync_claude_skills.py --prune --dry-run
python3 scripts/sync_claude_skills.py --prune    # or: make skills
```

The script produces links such as:

```text
.claude/skills/kg-anomaly-pipeline -> ../../.agents/skills/anomaly/kg-anomaly-pipeline
```

Edit the originals here, not the links. Run the sync after adding, moving, renaming or deleting a
skill. The script validates name collisions before creating anything, refuses to overwrite real
files and directories, and with `--prune` removes only broken links that resolve inside this
project's `.agents/skills/`.

`scripts/sync_claude_skills.py` is vendored unmodified from KGB-CIA. Fixes belong upstream
first, so the two copies do not diverge.
