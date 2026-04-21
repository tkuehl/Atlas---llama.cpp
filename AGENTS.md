# Instructions for this fork

This is a **private research fork** of [llama.cpp](https://github.com/ggml-org/llama.cpp).
It exists for personal experimentation, learning, and prototyping — changes
here are **not** destined for upstream and do not need to satisfy the
upstream project's contribution policy.

AI assistance is welcome for any purpose here: exploration, prototyping,
documentation, kernel sketches, quantization experiments, reading through
unfamiliar parts of the codebase, etc. The usual engineering discipline
still applies — understand what you're landing, keep commits scoped, don't
break the build — but there are no restrictions on how the code gets
written.

If something from this fork ever does need to go upstream, that change
would need to be reworked by a human contributor under the upstream
project's rules (see `CONTRIBUTING.md`). Treat any upstream-bound work as
a separate project.

## Useful Resources

Load as needed:

- [CONTRIBUTING.md](CONTRIBUTING.md) — upstream contribution rules (for reference only)
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md)
- [PEG parser](docs/development/parsing.md) — alternative to regex that llama.cpp uses to parse model output
- [Auto parser](docs/autoparser.md) — higher-level parser that uses PEG under the hood
- [Jinja engine](common/jinja/README.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [Research notes: low-bit quantization papers](docs/research/quantization-papers-2025.md)
