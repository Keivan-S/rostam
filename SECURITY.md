# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities **privately**. Do **not** open a public
GitHub issue for security problems.

Email **security@rostamlabs.com** with:

- A description of the issue and its impact.
- Steps to reproduce (proof-of-concept if possible).
- Affected version(s) / commit.

We aim to acknowledge reports within 3 business days and to provide a remediation
timeline after triage. Please give us a reasonable opportunity to release a fix
before any public disclosure.

## Scope

Rostam is a vector-database engine. Of particular interest:

- Memory-safety issues in the index, quantization (SIMD/unsafe), or CUDA paths.
- Sandbox-escape or resource-exhaustion in the WASM stored-procedure runtime.
- Authentication/authorization bypass (JWT/RBAC), or tenant-isolation breaks.
- Crash/DoS reachable from the network API (TCP/HTTP/gRPC) or from
  attacker-controlled snapshot/WAL/object-store input.
- Cryptographic issues in the object-store (SigV4) signing path.

## Supported Versions

Security fixes are provided for the latest released version. Older versions are
addressed on a best-effort basis.
