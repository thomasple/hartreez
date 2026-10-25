# Releasing to PyPI

Releases are started manually from an existing version tag. The release workflow
builds and validates one wheel and source distribution, uploads those files to
TestPyPI, and then uploads the same files to PyPI after the `pypi` environment
gate has passed.

## One-time GitHub setup

Create two GitHub environments in the repository settings:

- `testpypi`
- `pypi`

Configure a PyPI Trusted Publisher for each environment. Use owner
`thomasple`, repository `hartreez`, workflow filename `release.yml`, and set
the environment to the matching name (`testpypi` or `pypi`). No API token is
needed. In the `pypi` GitHub environment, require yourself as a reviewer and
leave **Prevent self-review** unchecked. This pauses publishing so you can
inspect the TestPyPI result and then approve it.

## Release steps

1. Merge the release workflow and packaging changes into `main`. Update the
   project version in `pyproject.toml` when needed, and rename `## Unreleased`
   in `CHANGELOG.md` to that version. Then tag and push the release commit.
   For the first release, the version and tag are `0.1.0` and `v0.1.0`.
2. Start the workflow for the tag, for example:

   ```sh
   gh workflow run release.yml --ref v0.1.0 -f release-tag=v0.1.0
   ```

   Replace both tag values for later releases.
3. Confirm the build and TestPyPI publishing jobs succeed. Install the release
   from TestPyPI in a clean environment and inspect its project page before
   allowing the `pypi` environment gate to continue.
4. Approve the `pypi` environment. The final job uploads the same
   distributions validated in the build job.

If a later job fails after TestPyPI succeeds, use GitHub's **Re-run failed
jobs** option so the already successful TestPyPI upload is not repeated.

The workflow checks that it was dispatched for the named tag and that the tag
matches the version in `pyproject.toml`. A version already uploaded to either
index cannot be reused; create a new version for a corrected release.
