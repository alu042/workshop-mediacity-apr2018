```
conda env update 
source activate mediacity

pip install --upgrade pip
pip install tensorflow-gpu==1.5 # Old version because Paperspace CPU lacks AVX instructions

pip install keras
pip install git+https://www.github.com/keras-team/keras-contrib.git

python -m ipykernel install --user --name mediacity --display-name "Mediacity"
jupyter labextension install @jupyter-widgets/jupyterlab-manager

sudo apt install tmux htop emacs tree
```
