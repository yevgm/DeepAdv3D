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
#     Command: bash ./requirements.txt
# [5] Install support for \\gip-main\data on Z: and \\132.68.39.11\gipfs on R:  # 132.68.39.13
#---------------------------------------------------------------------------------------------#
#                                       	
#---------------------------------------------------------------------------------------------#
# Opens and activates a conda env for Python 3.7.6 named DeepShape
git config --global user.name "Man Yevgeniy"
git config --global user.email "yevgenimen@camous.technion.ac.il"
git remote set-url origin https://github.com/yevgm/DeepAdv3D
eval "$(conda shell.bash hook)"
conda update -y -n base -c defaults conda
conda create -y -n DeepAdv3D python=3.7.0
conda activate DeepAdv3D
#---------------------------------------------------------------------------------------------#
#                                     	Primary Dependencies
#---------------------------------------------------------------------------------------------#
# Installs torch for Python 3.7 + Cuda 10.2 and Pytorch Geometric 
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
pip install torch-scatter==1.4+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==0.4.3+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==1.4.5+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric==1.3.2
# pip install tensorflow # Optional 
pip install tensorboard
# pip install pytorch-lightning
# pip install test-tube

# Primary 3D Geometry Modules:  
# pip install trimesh
# pip install scikit_learn
# pip install probreg
# pip install torchgeometry #consider also kornia
# pip install scipy # Installed by tensorflow automatically 

# Primary Visualizers:
pip install pyvista     
pip install matplotlib
pip install seaborn

# Utilities: 
pip install networkx
pip install numpy
pip install scipy
pip install yagmail
pip install meshio
pip install tqdm
pip install plyfile


conda install -y -c open3d-admin open3d # Last due to problems
# TO REMOVE: 
# gdist
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
