# Medical Image Processing
This is the repository for the WAT.ai Medical Image Processing Project for the 2022/2023 project season

## Setup
1. Clone the repository to your local machine: `git clone https://github.com/WAT-ai/medical-image-processing.git`
2. Install project dependencies: `pip install -r requirements.txt` 
3. Ensure you are working in the appropriate branch:
    - Check current branch: `git branch`
    - Switch branch: `git checkout [branch name]`
    - Create new branch: `git checkout -b [new branch name]`

## Data Handling and Version Control
1. Do not add changes to the data files within your commits
2. Github will not allow you to push the dataset into the repository (too large of a file size)
3. To avoid this issue, place your local dataset into a `data/` folder locally. The `.gitignore` file in this repository will prevent you from pushing anything within the data folder. 
    - Add changes: `git add [filename]` or `git add .` (adds all updated files)
    - Commit changes: `git commit -m "[commit details]"`
    - Push changes: `git push origin [your branch]` (ensure you are in the correct branch before pushing)
