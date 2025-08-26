# Installation

This guide walks you through installing the Histopath library and its required binary dependencies for working with whole-slide images (WSI).

!!! note "Python Requirement"
    You need Python 3.11 or newer to run histopath.


## Binary Dependencies

### OpenSlide

[OpenSlide](https://openslide.org) is required to read most WSI formats. You must install the native library before (or alongside) the Python package.

#### Linux (Debian / Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools
```

#### macOS (Homebrew)

```bash
brew update
brew install openslide
```

#### Windows

Download and install the [OpenSlide binaries](https://openslide.org/download/).


??? tip "Alternative: Python wheels"

    If you cannot install system packages, you can try using `openslide-bin` from PyPI.

    ```bash
    pip install openslide-bin
    ```

    !!! warning
        This method bundles the OpenSlide library and is **not compatible** with `pyvips`. Avoid using them together.



## Install the Python Package

Choose your preferred package manager to install the library.

=== "uv"

    ```bash
    uv add histopath
    ```

=== "pip"

    ```bash
    pip install histopath
    ```

=== "pdm"

    ```bash
    pdm add histopath
    ```