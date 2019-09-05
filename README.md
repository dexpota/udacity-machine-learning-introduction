[ https://github.com/JamshedVesuna/vim-markdown-preview#requireme
ntsminiconda-installation]: https://docs.conda.io/en/latest/miniconda.html

[license-cc]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[license-cc-badge]: https://img.shields.io/badge/License-CC%20BY--NC--SA-green.svg?style=for-the-badge

[license-mit]: https://opensource.org/licenses/MIT
[license-mit-badge]:  https://img.shields.io/github/license/dexpota/kitty-themes.svg?style=for-the-badge

# Machine Learning Introduction

> These notes are based on the Udacity course.

[![Notes licesing badge][license-cc-badge]][license-cc]
[![Code licesing badge][license-mit-badge]][license-mit]

## Prerequisites

### Miniconda Installation on Ubuntu

1. Download the installation script from the official website:
  ```bash
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  ```
2. Execute the installation script and follow the procedure:
  ```bash
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh
  ```
3. Recreate the conda environment:
  ```bash
  conda env create --name udacity-ml --file environments.yml
  ```

You can find more information on how to install miniconda on your system
[here](miniconda-installation).
