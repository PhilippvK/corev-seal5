# Seal5

This repository contains tools for our "**Se**mi-**a**utomated **L**LVM Generator for Custom RISC-**V** Instructions".

*Warning:* Reafactorings are still ongoing, hence several fewtures are currently missing.

## Prerequisites

### Submodules

Make sure to clone this repository using the `--recursive` flag or alternatively run `git submodule update --init --recursive` before running the scripts.

### Compatibility

The flow was only tested on Ubuntu 20.04 using Python v3.8. The required APT packages are as follow: `TODO`

### Python Environment

It is recommended to setup a virtual python environment for using seal5. This can be accomplished as follows:

```python
python3 -m venv venv
source venv/bin/activate
```

## Examples

Examples are provided to demonstrate the end-to-end flow for different types of custom instructions in a step-by-step fashion.

List of examples:

- **XCoreVMac**: See `examples/xcorevmac.sh`

## Preview (WIP)

After the refactoring is completed, the flow should be as straightforward as:

```bash
seal5 init ./llvm-project  # Add markers to codebase, configure and compile
seal5 load ./XCVMac.core_desc yaml/XCVMac.yml  # Load coredsl and config files
seal5 transform [--fail-on-error] ...  # Analyze instructions, extract side-effects,...
seal5 generate [--skip-build] [--no-patterns] ...  # Generate patches, apply to repo, rebuild llvm (includes intermediate llvm build if patterns are generated)
seal5 deploy [--no-combine] [--msg MSG] ...  # Combine patches to single patchset, verify functionality
seal5 export ./out.zip  # Archive patches, logs and reports
```

To have more flexibility regarding the generated code for specific extensions or individual instructions, the patches to be generated are configured using YAML files.
