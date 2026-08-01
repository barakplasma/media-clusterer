## ❌[MegaLinter](https://megalinter.io/9.6.0) analysis: Error



| Descriptor  |                                               Linter                                                |Files|Fixed|Errors|Warnings|Elapsed time|
|-------------|-----------------------------------------------------------------------------------------------------|----:|----:|-----:|-------:|-----------:|
|✅ ACTION    |[actionlint](https://megalinter.io/9.6.0/descriptors/action_actionlint)                              |    1|     |     0|       0|       0.25s|
|✅ ACTION    |[zizmor](https://megalinter.io/9.6.0/descriptors/action_zizmor)                                      |    1|    0|     0|       0|       0.53s|
|✅ MARKDOWN  |[markdownlint](https://megalinter.io/9.6.0/descriptors/markdown_markdownlint)                        |    5|    1|     0|       0|       1.17s|
|✅ MARKDOWN  |[markdown-table-formatter](https://megalinter.io/9.6.0/descriptors/markdown_markdown_table_formatter)|    5|    1|     0|       0|       0.35s|
|✅ REPOSITORY|[betterleaks](https://megalinter.io/9.6.0/descriptors/repository_betterleaks)                        |  yes|     |    no|      no|        1.0s|
|✅ REPOSITORY|[checkov](https://megalinter.io/9.6.0/descriptors/repository_checkov)                                |  yes|     |    no|      no|      18.75s|
|❌ REPOSITORY|[gitleaks](https://megalinter.io/9.6.0/descriptors/repository_gitleaks)                              |  yes|     |     1|      no|        0.9s|
|✅ REPOSITORY|[git_diff](https://megalinter.io/9.6.0/descriptors/repository_git_diff)                              |  yes|     |    no|      no|       0.01s|
|❌ REPOSITORY|[grype](https://megalinter.io/9.6.0/descriptors/repository_grype)                                    |  yes|     |     2|      no|      61.25s|
|❌ REPOSITORY|[osv-scanner](https://megalinter.io/9.6.0/descriptors/repository_osv_scanner)                        |  yes|     |     2|      no|       0.85s|
|✅ REPOSITORY|[secretlint](https://megalinter.io/9.6.0/descriptors/repository_secretlint)                          |  yes|     |    no|      no|       0.97s|
|✅ REPOSITORY|[syft](https://megalinter.io/9.6.0/descriptors/repository_syft)                                      |  yes|     |    no|      no|       2.43s|
|❌ REPOSITORY|[trivy](https://megalinter.io/9.6.0/descriptors/repository_trivy)                                    |  yes|     |     1|      no|       11.2s|
|✅ REPOSITORY|[trivy-sbom](https://megalinter.io/9.6.0/descriptors/repository_trivy_sbom)                          |  yes|     |    no|      no|       1.92s|
|✅ REPOSITORY|[trufflehog](https://megalinter.io/9.6.0/descriptors/repository_trufflehog)                          |  yes|     |    no|      no|       3.52s|
|✅ YAML      |[prettier](https://megalinter.io/9.6.0/descriptors/yaml_prettier)                                    |    1|    0|     0|       0|       0.44s|
|✅ YAML      |[v8r](https://megalinter.io/9.6.0/descriptors/yaml_v8r)                                              |    1|     |     0|       0|       2.58s|
|✅ YAML      |[yamllint](https://megalinter.io/9.6.0/descriptors/yaml_yamllint)                                    |    1|     |     0|       0|       0.61s|

## Detailed Issues

<details>
<summary>❌ REPOSITORY / gitleaks - 1 error</summary>

```
○
    │╲
    │ ○
    ○ ░
    ░    gitleaks

Finding:     const UNSPLASH_ACCESS_KEY = 'REDACTED'
Secret:      REDACTED
RuleID:      generic-api-key
Entropy:     4.990105
File:        src/app.ts
Line:        56
Commit:      HIDDEN_BY_MEGALINTERAuthor:      Michael Salaverry
Email:       barakplasma@gmail.com
Date:        2026-04-06T06:50:14Z
Fingerprint: ddc99343d75ec378206eecb16d94d11787cae039:src/app.ts:generic-api-key:56
Link:        https://github.com/barakplasma/media-clusterer/blob/ddc99343d75ec378206eecb16d94d11787cae039/src/app.ts#L56

10:11AM INF 189 commits scanned.
10:11AM INF scanned ~1267918 bytes (1.27 MB) in 848ms
10:11AM WRN leaks found: 1
```

</details>

<details>
<summary>❌ REPOSITORY / grype - 2 errors</summary>

```
[0000]  WARN no explicit name and version provided for directory source, deriving artifact ID from the given path (which is not ideal) from=syft
NAME     INSTALLED  FIXED IN  TYPE  VULNERABILITY        SEVERITY  EPSS         RISK  
adm-zip  0.5.17     0.6.0     npm   GHSA-xcpc-8h2w-3j85  High      0.4% (35th)  0.3   
sharp    0.34.5     0.35.0    npm   GHSA-f88m-g3jw-g9cj  High      N/A          N/A
[0061] ERROR discovered vulnerabilities at or above the severity threshold
```

</details>

<details>
<summary>❌ REPOSITORY / osv-scanner - 2 errors</summary>

```
Scanning dir .
Starting filesystem walk for root: /
Scanned package-lock.json file and found 278 packages
End status: 32 dirs visited, 132 inodes visited, 1 Extract calls, 20.784651ms elapsed, 20.784862ms wall time

Total 2 packages affected by 2 known vulnerabilities (0 Critical, 2 High, 0 Medium, 0 Low, 0 Unknown) from 1 ecosystem.
2 vulnerabilities can be fixed.

+-------------------------------------+------+-----------+---------+---------+---------------+-------------------+
| OSV URL                             | CVSS | ECOSYSTEM | PACKAGE | VERSION | FIXED VERSION | SOURCE            |
+-------------------------------------+------+-----------+---------+---------+---------------+-------------------+
| https://osv.dev/GHSA-xcpc-8h2w-3j85 | 7.5  | npm       | adm-zip | 0.5.17  | 0.6.0         | package-lock.json |
| https://osv.dev/GHSA-f88m-g3jw-g9cj | 7.0  | npm       | sharp   | 0.34.5  | 0.35.0        | package-lock.json |
+-------------------------------------+------+-----------+---------+---------+---------------+-------------------+
```

</details>

<details>
<summary>❌ REPOSITORY / trivy - 1 error</summary>

```
2026-08-01T10:11:08Z	INFO	[vulndb] Need to update DB
2026-08-01T10:11:08Z	INFO	[vulndb] Downloading vulnerability DB...
2026-08-01T10:11:08Z	INFO	[vulndb] Downloading artifact...	repo="mirror.gcr.io/aquasec/trivy-db:2"
28.33 MiB / 103.26 MiB [---------------->___________________________________________] 27.43% ? p/s ?61.42 MiB / 103.26 MiB [----------------------------------->________________________] 59.48% ? p/s ?91.95 MiB / 103.26 MiB [----------------------------------------------------->______] 89.05% ? p/s ?103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 124.88 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 124.88 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 124.88 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 116.82 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 116.82 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 116.82 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 109.28 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 109.28 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 109.28 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 102.23 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 102.23 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [------------------------------------------->] 100.00% 102.23 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 95.64 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 95.64 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 95.64 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 89.47 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 89.47 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 89.47 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 83.70 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 83.70 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 83.70 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 78.30 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 78.30 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 78.30 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 73.24 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 73.24 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-------------------------------------------->] 100.00% 73.24 MiB p/s ETA 0s103.26 MiB / 103.26 MiB [-----------------------------------------------] 100.00% 17.25 MiB p/s 6.2s2026-08-01T10:11:15Z	INFO	[vulndb] Artifact successfully downloaded	repo="mirror.gcr.io/aquasec/trivy-db:2"
2026-08-01T10:11:16Z	INFO	[vuln] Vulnerability scanning is enabled
2026-08-01T10:11:16Z	INFO	[misconfig] Misconfiguration scanning is enabled
2026-08-01T10:11:16Z	INFO	[checks-client] Need to update the checks bundle
2026-08-01T10:11:16Z	INFO	[checks-client] Downloading the checks bundle...
234.65 KiB / 234.65 KiB [------------------------------------------------------] 100.00% ? p/s 100ms2026-08-01T10:11:19Z	INFO	[npm] Run "npm install" to collect the license information of packages	dir="node_modules"
2026-08-01T10:11:19Z	INFO	Suppressing dependencies for development and testing. To display them, try the '--include-dev-deps' flag.
2026-08-01T10:11:19Z	INFO	Number of language-specific files	num=1
2026-08-01T10:11:19Z	INFO	[npm] Detecting vulnerabilities...
2026-08-01T10:11:19Z	INFO	Detected config files	num=0

Report Summary

┌───────────────────┬──────┬─────────────────┬───────────────────┐
│      Target       │ Type │ Vulnerabilities │ Misconfigurations │
├───────────────────┼──────┼─────────────────┼───────────────────┤
│ package-lock.json │ npm  │        2        │         -         │
└───────────────────┴──────┴─────────────────┴───────────────────┘
Legend:
- '-': Not scanned
- '0': Clean (no security findings detected)


For OSS Maintainers: VEX Notice
--------------------------------
If you're an OSS maintainer and Trivy has detected vulnerabilities in your project that you believe are not actually exploitable, consider issuing a VEX (Vulnerability Exploitability eXchange) statement.
VEX allows you to communicate the actual status of vulnerabilities in your project, improving security transparency and reducing false positives for your users.
Learn more and start using VEX: https://trivy.dev/docs/v0.71/guide/supply-chain/vex/repo#publishing-vex-documents

To disable this notice, set the TRIVY_DISABLE_VEX_NOTICE environment variable.


package-lock.json (npm)
=======================
Total: 2 (UNKNOWN: 0, LOW: 0, MEDIUM: 0, HIGH: 2, CRITICAL: 0)

┌─────────┬─────────────────────┬──────────┬────────┬───────────────────┬───────────────┬─────────────────────────────────────────────────────────────┐
│ Library │    Vulnerability    │ Severity │ Status │ Installed Version │ Fixed Version │                            Title                            │
├─────────┼─────────────────────┼──────────┼────────┼───────────────────┼───────────────┼─────────────────────────────────────────────────────────────┤
│ adm-zip │ CVE-2026-39244      │ HIGH     │ fixed  │ 0.5.17            │ 0.6.0         │ adm-zip: adm-zip: Denial of Service via crafted ZIP file    │
│         │                     │          │        │                   │               │ leading to excessive...                                     │
│         │                     │          │        │                   │               │ https://avd.aquasec.com/nvd/cve-2026-39244                  │
├─────────┼─────────────────────┤          │        ├───────────────────┼───────────────┼─────────────────────────────────────────────────────────────┤
│ sharp   │ GHSA-f88m-g3jw-g9cj │          │        │ 0.34.5            │ 0.35.0        │ sharp inherited vulnerabilities in libvips: CVE-2026-33327, │
│         │                     │          │        │                   │               │ CVE-2026-33328, CVE-2026-35590, CVE-2026-35591              │
│         │                     │          │        │                   │               │ https://github.com/advisories/GHSA-f88m-g3jw-g9cj           │
└─────────┴─────────────────────┴──────────┴────────┴───────────────────┴───────────────┴─────────────────────────────────────────────────────────────┘

📣 Notices:
  - Version 0.72.0 of Trivy is now available, current version is 0.71.2

To suppress version checks, run Trivy scans with the --skip-version-check flag
```

</details>


### Notices

📣 **MegaLinter 9.5.0 is out!** Discover the new features and security recommendations in the [release announcement](https://github.com/oxsecurity/megalinter/issues/7835). (Skip this info by defining `SECURITY_SUGGESTIONS: false`)

See detailed reports in MegaLinter artifacts
_Set `VALIDATE_ALL_CODEBASE: true` in mega-linter.yml to validate all sources, not only the diff_

[![MegaLinter is graciously provided by OX Security](https://raw.githubusercontent.com/oxsecurity/megalinter/main/docs/assets/images/ox-banner.png)](https://www.ox.security/?ref=megalinter)
Show us your support by [**starring ⭐ the repository**](https://github.com/oxsecurity/megalinter)