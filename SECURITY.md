# Security Policy

## Supported versions

| Version | Security fixes |
|---------|----------------|
| 0.9.x   | Yes            |
| < 0.9   | No             |

Security fixes are released as patch versions of the latest minor release and
are announced in the
[GitHub security advisories](https://github.com/Nitidio/nitid/security/advisories)
and the [CHANGELOG](CHANGELOG.md).

## Reporting a vulnerability

Please do not open a public GitHub issue, discussion or pull request for a
security vulnerability.

Report it privately through either channel:

- GitHub: [report a vulnerability](https://github.com/Nitidio/nitid/security/advisories/new)
  (private vulnerability reporting).
- Email: **security@vaelsys.com**.

Include, as far as you can:

- the affected version or commit, and how nitid was installed;
- a description of the vulnerability and its impact;
- steps or a minimal proof of concept to reproduce it.

## What happens next

| Step | Target |
|------|--------|
| Acknowledgement of your report | within 3 working days |
| Initial assessment: confirmed or not, and severity | within 10 working days |
| Fix or mitigation for a confirmed vulnerability | within 90 days of the report |

We keep you informed of progress and agree the disclosure date with you.

## Disclosure policy

We follow coordinated disclosure:

1. We confirm the vulnerability and prepare a fix privately.
2. We publish the fix and a GitHub security advisory at the same time, and
   request a CVE identifier where applicable.
3. We credit the reporter in the advisory unless they prefer not to be named.

If a fix is not available within 90 days, we agree with the reporter on either
a short extension or disclosure with the available mitigations. If a
vulnerability is being actively exploited, we may disclose earlier.

## Scope

In scope: the code in this repository and the packages published from it on
PyPI.

Out of scope, please report upstream:

- vulnerabilities in third-party dependencies such as PyTorch, OpenVINO or
  OpenCV, unless nitid uses them in an unsafe way;
- the pretrained weights and datasets that nitid downloads, which third parties
  publish.

nitid checkpoints are PyTorch pickle files and are loaded with
`torch.load(..., weights_only=False)`. Loading a checkpoint can therefore run
arbitrary code: only load checkpoints from sources you trust. Safer loading
is tracked in [#27](https://github.com/Nitidio/nitid/issues/27). Reports that
depend only on loading a malicious checkpoint are known behaviour, not a new
vulnerability.

## Safe harbour

We will not pursue legal action against research carried out in good faith
under this policy: accessing only your own data and systems, avoiding privacy
violations and service disruption, and giving us reasonable time to fix the
issue before disclosure.
