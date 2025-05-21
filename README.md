# TeamDN1

## Steps to run the code in this repo (for supervisor/testing use)

- Create a conda environment which uses Python 3.12 (e.g. `conda create -n team-dn1-project python=3.12`)

- Run `pip install -r requirements.txt`

- Something to note is that this installs the CPU-bound version of Pytorch. Pip freeze doesn't like the CUDA version and therefore will not create the correct requirements doc if this version is used. I would highly recommend adding this version to your environment separately, using the command `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` for Windows or `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` for Linux (not available on Mac)

- Next we will need to get the fasttext embeddings to correctly run the supervised models. To do this, head to the fasttext folder and run the command `curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz` or `wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz` depending on your system or preference

- Extract the embeddings into the fasttext folder. You should be left with a file called `cc.en.300.vec`

- Head back to the root directory of this project and run your desired model. Models must be run from the root (`com6911-teamdn1`) due to the paths mentioned in their code. For example, if I want to run the fnn model, this can be done with `python supervised_model/fnn/alessandro_fnn.py`

Any further questions/help please contact me (`aperelli1@sheffield.ac.uk`). Happy testing!

## For editing (team-use)

### Protected branch

For this repo, the main branch is protected. This means that work cannot be directly pushed onto it without a push request (which needs to be reviewed by two people). In order to do this follow the steps:

- Create a branch for the work you are doing. This could be `cnn-model` etc. Branches can be created using the command `git checkout -b name-of-branch`

- Push your work to this branch

- Go to the pull requests section of this repo and create a pull request to merge your branch into main (make sure it's into main)

- Wait for review approval

- Merge branch

Any questions regarding this, please let me know (Alessandro)

### Issues

Github issues is basically a task tool. You create an issue (task) and then assign it to someone to complete. I have created the ones for our classifier team as examples.

### Repo rules

- One issue per week per person - please try to keep to this, as to ensure equal and fair allocation of work

- Editing other people's work - I'm sure this will not happen, but please refrain from overriding the work of others especially cross sub-teams (knn and classifier). Suggestions are more than welcome (of course) but please let the sub-team know and allow someone from that team to implement it. This ensures fair allocation of work.

Any questions regarding anything about this repo, let me know (Alessandro)

Happy coding!
