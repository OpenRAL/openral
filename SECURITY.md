# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `master` | Yes |
| Releases | Latest minor only |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Preferred: use GitHub's [private vulnerability reporting](https://github.com/OpenRAL/openral/security/advisories/new)
("Report a vulnerability" under the repository **Security** tab), which keeps the
fix and coordinated disclosure in one private place.

Alternatively, email: security@openral.com

Include:
- Description of the vulnerability.
- Steps to reproduce.
- Potential impact.
- Any suggested fix.

We will respond within 48 hours and work with you on coordinated disclosure.

## Safety-critical issues

For issues involving physical safety of robots or people (E-stop bypass, actuation path bugs,
safety kernel defects), email security@openral.com with "[safety]" in the subject line before
using the [Safety issue template](.github/ISSUE_TEMPLATE/safety.yml). These are treated as
highest priority.
