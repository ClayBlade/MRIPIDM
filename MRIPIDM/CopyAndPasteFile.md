Make Dir:

mkdir /root/output

cd output

mkdir /root/output/labeledSpaceJSON





Scp:

scp -i C:\\Users\\clayt\\.ssh\\id\_rsa -P \[port] (-r) \[file/(directory)] root@\[ip]:\[destination]





git clone:

git clone -b master https://github.com/ClayBlade/MRIPIDM





Installing nvcc:

sudo apt update

sudo apt install build-essential

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86\_64/cuda-keyring\_1.1-1\_all.deb

sudo dpkg -i cuda-keyring\_1.1-1\_all.deb

sudo apt-get update

sudo apt-get install -y cuda-toolkit-12-4 \[**nvidia-smi and nvcc --version have to match**]

export PATH=/usr/local/cuda-12.1/bin:$PATH

export LD\_LIBRARY\_PATH=/usr/local/cuda-12.1/lib64:$LD\_\_LIRBRARY\_PATH





Installing ipps:

wget https://registrationcenter-download.intel.com/akdlm/IRC\_NAS/d9649232-67ed-489e-8cd8-2c4c54b06135/intel-ipp-2022.2.0.583\_offline.sh

sudo sh ./intel-ipp-2022.2.0.583\_offline.sh

chmod +x intel-ipp-2022.2.0.583\_offline.sh

sudo apt -y install ncurses-term

sudo sh ./intel-ipp-2022.2.0.583\_offline.sh -a --silent --eula accept

export IPP\_ROOT=/opt/intel/ipp\_2022.2.0

export CPATH=$IPP\_ROOT/include:$CPATH

export LD\_LIBRARY\_PATH=$IPP\_ROOT/lib:$LD\_LIBRARY\_PATH





Activate Environment

source /opt/intel/oneapi/setvars.sh





Test change:

cd /root

rm -r MRIPIDM

git clone -b master https://github.com/ClayBlade/MRIPIDM 

cd MRIPIDM/MRIPIDM

nvcc -I/opt/intel/oneapi/ipp/2022.2/include/ipp DoScanAtGPU.cu helperFuncs.cpp -o Bloch -L/opt/intel/oneapi/ipp/latest/lib/intel64 -lipps -lippcore -Xcompiler -fopenmp



upload to git:

git add MRIPIDM

git commit -am"\[some message]"

git push https://github.com/ClayBlade/MRIPIDM HEAD:master

