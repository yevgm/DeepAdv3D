#!/bin/bash
#---------------------------------------------------------------------------------------------#
#                                       		Prep
#---------------------------------------------------------------------------------------------#
# [1] Download and install Anaconda + PyCharm/other IDEs
#     * If anaconda is already installed, make sure it is in the PATH variable 
#       - in some versions, add only condabin 
#       - in some versions, add both conda bin and Scripts in the anaconda directory 
# [2] For Windows: Install Git Bash 
# [3] Pull the relevant code from Github 
# [4] Run this script by opening up the GitBash shell in the code and running: 
#     Command: ./setup.sh
# 
#
# - if conda commands isn't recognised, you need to add to the PATH variable conda dir:
#   set PATH=%PATH%;"C:\Users\Anaconda3";"C:\Users\Anaconda3\Scripts" in powershell
# [5] Datasets:
#	    -FAUST: http://faust.is.tue.mpg.de/
# [6] create a folder datasets/faust/raw <- put here the test data

# [7] open tensorboard:
# cd <main location of tensor_board_logs>
# tensorboard --logdir=tensor_board_logs
#---------------------------------------------------------------------------------------------#
#                                       		Usefull commands
#---------------------------------------------------------------------------------------------#
# conda env remove -n DeepAdv3D
#---------------------------------------------------------------------------------------------#
#                                       	
#---------------------------------------------------------------------------------------------#
# add to path conda, pip, python, tensorboard. Create new container for faust dataset
# mkdir -p parentfolder/{subfolder1,subfolder2,subfolder3} for multiple subfolders
export PATH=$PATH:"/c/Users/`whoami`/AppData/Local/Continuum/anaconda3"
export PATH=$PATH:"/c/Users/`whoami`/AppData/Local/Continuum/anaconda3/Scripts"

mkdir -p "../model_data"
#---------------------------------------------------------------------------------------------#
#                                       	
#---------------------------------------------------------------------------------------------#
# Opens and activates a conda env for Python 3.7.6 named 
git config --global user.name "Man Yevgeniy"
git config --global user.email "yevgenimen@campus.technion.ac.il"
git remote set-url origin https://github.com/yevgm/DeepAdv3D
eval "$(conda shell.bash hook)"
conda update -y -n base -c defaults conda
conda create -y -n DeepAdv3D python=3.7.0
conda activate DeepAdv3D
#---------------------------------------------------------------------------------------------#
#                                     	Primary Dependencies
#---------------------------------------------------------------------------------------------#
# Installs torch for Python 3.7 + Cuda 10.2 and Pytorch Geometric 
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -y -c pytorch


# Primary 3D Geometry Modules:  
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
# pip install torch-geometric==1.3.2

# pip install tensorflow # Optional 
pip install tensorboard
pip install tb-nightly
conda install -y -c conda​-forge tensorboard 

# add tensorboard to path
export PATH=$PATH:"`python tensor_board.py`"
echo $PATH

# Primary Visualizers:
pip install pyvista    
pip install vtk==8.1.2 
pip install matplotlib
pip install seaborn

# Utilities: 
pip install networkx
# some windows like this
pip install numpy==1.18.5
pip install scipy==1.5.0
# some this
# pip install numpy
# pip install scipy
pip install yagmail
pip install meshio
pip install openmesh
pip install tqdm
pip install plyfile
pip install sklearn

conda install -y -c open3d-admin open3d # Last due to problems

#---------------------------------------------------------------------------------------------#
#                            			Donwload Data
#---------------------------------------------------------------------------------------------#
python ./src/utils/set_work_env/download_data.py

#---------------------------------------------------------------------------------------------#
#                            			Collaterals
#---------------------------------------------------------------------------------------------#
# [*] Optional: If you want to support full tensoboard features, install tensorflow via:
#     Command: pip install tensorflow
#
# [*] Optional: Some very specific features require point_cloud_utils by fwilliams: 
#      pip install git+git://github.com/fwilliams/point-cloud-utils
#      or conda install -c conda-forge point_cloud_utils
#
# [*] GMAIl Credentials - We use the yagmail package to send the Tensorboard logs to
#      a shared email account. You can configure the email account by placing a txt
#      file under data/collaterals/gmail_credentials.txt with the following information:
#      user=yourmail@gmail.com
#      pass=yourpassword
#---------------------------------------------------------------------------------------------#
#                            	Collaterals - Windows only
#---------------------------------------------------------------------------------------------#
# [*]  Optional: In order to support nn.visualize(), please install Graphviz
#  	   *  Surf to: https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi
#      *  Install the application to C:\Program Files (x86)\Graphviz2.38
#      *  Add to your PATH variable: "C:\Program Files (x86)\Graphviz2.38\bin"
#---------------------------------------------------------------------------------------------#
#                			Important Notes (to save painful debugging)
#---------------------------------------------------------------------------------------------#
# [*]  PyRender support for the Azimuthal Projections: 
#	   *  Location: src/data/prep
#      *  Usage: Only under Linux. Please see prep_main.py for more details
#      *  Compilation: executable is supplied, but you might need to compile it again on 
#        a different machine. Remember to fix the GLM include path in compile.sh
#        CUDA runtime 10.2 must be installed to enable compilation. We recommend the 
#        deb(local) setting. 
#         
# [*] PyCharm Users: Make sure that 'src/core' is enabled as the Source Root:
#---------------------------------------------------------------------------------------------#
#
#---------------------------------------------------------------------------------------------#