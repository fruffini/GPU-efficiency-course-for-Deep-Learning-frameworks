<div align="left" style="position: relative;">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="right" width="30%" style="margin: -20px 0 0 20px;">
<h1>GPU-EFFICIENCY-COURSE-FOR-DEEP-LEARNING-FRAMEWORKS</h1>
<p align="left">
	<em>A hands-on course to optimize GPU usage for deep learning frameworks, focusing on medical imaging tasks.</em>
</p>
<p align="left">
	<img src="https://img.shields.io/github/license/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks?style=for-the-badge&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks?style=for-the-badge&logo=git&logoColor=white&color=0080ff" alt="last-commit">

</p>
<p align="left">Built with the tools and technologies:</p>
<p align="left">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=for-the-badge&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Python">
</p>
</div>
<br clear="right">

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

This course is designed to teach practical techniques for improving GPU efficiency in deep learning workflows, with a focus on medical imaging tasks. It covers topics such as GPU monitoring, optimizing data pipelines, improving model performance, and maximizing GPU utilization. The course is structured into chapters, each containing scripts, exercises, and examples to help you apply the concepts in real-world scenarios.

---

##  Features

- Learn to monitor and track GPU usage effectively.
- Optimize data pipelines for large-scale medical imaging datasets.
- Improve model performance through precision tuning and scaling.
- Maximize GPU utilization for deep learning tasks.
- Hands-on exercises and scripts for practical learning.

---

##  Project Structure

```sh
└── GPU-efficiency-course-for-Deep-Learning-frameworks/
    ├── LICENSE
    ├── README.md
    ├── chapter0_track_gpus
    │   ├── 01_nvidia_smi_nvtop_basics.sh
    │   ├── 02_gpu_monitoring.py
    │   ├── 03_slurm_gpu_info.sh
    │   ├── 04_practical_exercises.sh
    │   └── Lesson_0.md
    ├── chapter1_improve_the_model_performance
    │   ├── 01_practical_exercise.sh
    │   ├── Lesson_1.md
    │   ├── compute_flops_efficiency.py
    │   ├── mock_precision_accumulation.py
    │   ├── mock_precision_benchmark.py
    │   ├── mock_training_opt.py
    │   └── mock_training_scaling.py
    ├── chapter2_data_efficiency
    │   ├── 01_practical_exercise.sh
    │   └── Lesson_2.md
    └── chapter3_maximise_GPU
        ├── 01_maximize_the_GPU_consumptions.sh
        └── Lesson_3.md
```


###  Project Index
<details open>
	<summary><b><code>GPU-EFFICIENCY-COURSE-FOR-DEEP-LEARNING-FRAMEWORKS/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b>Root directory</b></td>
				<td>Contains the main README, license, and project-level files.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- chapter0_track_gpus Submodule -->
		<summary><b>chapter0_track_gpus</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b>GPU Tracking</b></td>
				<td>Scripts and exercises for monitoring GPU usage and querying SLURM.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- chapter1_improve_the_model_performance Submodule -->
		<summary><b>chapter1_improve_the_model_performance</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b>Model Performance</b></td>
				<td>Learn to optimize model training through precision and scaling techniques.</td>
			</tr>
			</table>
		</blockquote>
	</details>
  <details> <!-- chapter2_data_efficiency Submodule -->
		<summary><b>chapter2_data_efficiency</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b>Data Efficiency</b></td>
				<td>Optimize data loading and prefetching for large datasets.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- chapter3_maximise_GPU Submodule -->
		<summary><b>chapter3_maximise_GPU</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b>Maximize GPU Utilization</b></td>
				<td>Techniques to ensure efficient GPU usage during training.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with GPU-efficiency-course-for-Deep-Learning-frameworks, ensure your runtime environment meets the following requirements:

- **Programming Language:** Shell [Linux Shell Course](https://github.com/rcgsheffield/linux-shell)


###  Installation

Install GPU-efficiency-course-for-Deep-Learning-frameworks using one of the following methods:

## 1. Clone the GPU-efficiency-course-for-Deep-Learning-frameworks repository:
```sh
❯ git clone https://github.com/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks
```

## 2. Navigate to the project directory:
```sh
❯ cd GPU-efficiency-course-for-Deep-Learning-frameworks
```

## 3.0 Alvis Users Only CheckList

> **Note**: These instructions are specifically for users of the Alvis HPC cluster.

### Prerequisites
- Upload your project to your directory on Alvis
- Ensure you have access to the required modules

---

## **[📚 Additional Examples for Alvis Usage](https://github.com/c3se/alvis-intro)**

**For comprehensive guides, tutorials, and additional examples on using Alvis, please visit the official Alvis Introduction repository above.**

---

### Setup Instructions

First, load the required modules:
```sh
# Navigate to your project directory
cd .../GPU_efficiency_course

# Clean the module environment
module purge

# Load required modules
module load Python/3.12.3-GCCcore-13.3.0
module load nvtop/3.2.0-GCCcore-13.3.0
module load virtualenv/20.26.2-GCCcore-13.3.0
```

### 3.1 Create Virtual Environment and Install Dependencies

It's recommended to create a virtual environment to manage your Python dependencies:
```sh
# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source --system-site-packages  venv/bin/activate

# Install the required dependencies
pip install pynvml psutil matplotlib torch monai
```

### Verify Installation

After installation, you can verify that the packages are installed correctly:
```sh
pip list
```

### Deactivating the Environment

When you're done working, deactivate the virtual environment:
```sh
deactivate
```


###  Usage
Run GPU-efficiency-course-for-Deep-Learning-frameworks using the following command:
echo 'INSERT-RUN-COMMAND-HERE'

###  Testing
Run the test suite using the following command:
echo 'INSERT-TEST-COMMAND-HERE'

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks/issues)**: Submit bugs found or log feature requests for the `GPU-efficiency-course-for-Deep-Learning-frameworks` project.
- **💡 [Submit Pull Requests](https://github.com/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=fruffini/GPU-efficiency-course-for-Deep-Learning-frameworks">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

