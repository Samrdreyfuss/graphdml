# Security policy

## Supported versions

Security fixes are made to the latest release on PyPI and to `main`.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private reporting:
go to the repository's **Security** tab and choose **Report a vulnerability**. Include what
you found, how to reproduce it, and the graphdml and Python versions.

You can expect an acknowledgement within a week. Confirmed issues are fixed in a new
release and credited in the changelog unless you prefer otherwise.

## Scope notes

graphdml runs locally and makes no network requests, except when you ask it to:
`make_lastfm_promo` downloads one checksum-verified file from SNAP, and the Neo4j module
connects to the server you configure. In the Neo4j module, labels, relationship types and
property names are validated and quoted, and all values are sent as query parameters. Never
commit database credentials; pass them through environment variables.
