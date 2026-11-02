# Security Policy

This is OpenRAL's coordinated vulnerability disclosure policy for this repository and
the packages, images and rSkills built from it.

## Supported versions

| Version | Supported |
|---|---|
| `master` | Yes |
| Releases | Latest minor only |

Security fixes ship in a patch release of the latest minor version.

## Reporting a vulnerability

**Do not open a public GitHub issue, discussion or pull request for security vulnerabilities.**

Preferred: use GitHub's [private vulnerability reporting](https://github.com/OpenRAL/openral/security/advisories/new)
("Report a vulnerability" under the repository **Security** tab). It keeps the
fix and the coordinated disclosure in one private place.

Alternatively, email: security@openral.com

Include:
- The affected version (release, commit or image tag).
- A description of the vulnerability.
- Steps to reproduce.
- Potential impact.
- Whether you know of it being exploited.
- Any suggested fix.

## What we commit to

| Step | Target |
|---|---|
| Acknowledge your report | 48 hours |
| Initial assessment and severity (CVSS) | 5 business days |
| Fix or mitigation for Critical / High | 30 days |
| Fix or mitigation for Medium / Low | 90 days, or the next release |
| Public disclosure | when a fix is available, by default no later than 90 days after your report. We agree the date with you. |

Once a fix is released, we publish a GitHub security advisory. It gives a description,
the affected versions, the impact, the severity and how to remediate, and requests a
CVE where applicable. We credit reporters unless they ask us not to.

If a vulnerability is in a third-party dependency, we report it to that project's
maintainers as well.

## Safety-critical issues

For issues involving physical safety of robots or people (E-stop bypass, actuation path bugs,
safety kernel defects), email security@openral.com with "[safety]" in the subject line before
using the [Safety issue template](.github/ISSUE_TEMPLATE/safety.yml). These are treated as
highest priority.
