# Release Checklist

- Add the final paper citation to `README.md`.
- Add the license approved by the authors' institution.
- Verify the repository contains no model weights, generated videos, logs, or
  local absolute paths.
- Run the four training CLIs with `--help` and all inference shell scripts with
  `DRY_RUN=1` in the release environment.
