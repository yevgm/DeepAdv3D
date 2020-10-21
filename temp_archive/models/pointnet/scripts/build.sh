SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
echo $SCRIPTPATH

g++ -std=c++11 "/home/jack/OneDrive/Studies/Undergrad_Project/Project_kickstart/Geometric Deep Learning Ramp/pointnet.pytorch-master/utils"/render_balls_so.cpp -o "/home/jack/OneDrive/Studies/Undergrad_Project/Project_kickstart/Geometric Deep Learning Ramp/pointnet.pytorch-master/utils"/render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
