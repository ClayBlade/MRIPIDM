Make Dir:

mkdir /root/output

cd output

mkdir /root/output/labeledSpaceJSON

mkdir /root/output/ParametricMaps

mkdir /root/output/ReconstructedMRI



Scp from local to remote:

scp -i C:\\Users\\clayt\\.ssh\\id\_rsa -P \[port] (-r) \[file/(directory)] root@\[ip]:\[destination]



Scp from remote to local:

scp -i C:\\Users\\clayt\\.ssh\\id\_rsa -P 38568 root@213.181.122.2:/root/output/ReconstructedMRI/mri\_result.png "D:\\Projects\\MRIPIDMoutput\\ReconstructedMRI"



git clone:

git clone -b master https://github.com/ClayBlade/MRIPIDM



Installing nvcc:

sudo apt update

sudo apt install build-essential

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86\_64/cuda-keyring\_1.1-1\_all.deb

sudo dpkg -i cuda-keyring\_1.1-1\_all.deb

sudo apt-get update

sudo apt-get install -y cuda-toolkit-12-4 \[\[Or whatever version nvidia-smi returns, I really don't know]]



Installing ipps:

wget https://registrationcenter-download.intel.com/akdlm/IRC\_NAS/d9649232-67ed-489e-8cd8-2c4c54b06135/intel-ipp-2022.2.0.583\_offline.sh

chmod +x intel-ipp-2022.2.0.583\_offline.sh

sudo ./intel-ipp-2022.2.0.583\_offline.sh

sudo sh ./intel-ipp-2022.2.0.583\_offline.sh -a --silent --eula accept





On restarting VM instance:

source /opt/intel/oneapi/setvars.sh



Test change:

**cd /root**

**rm -r MRIPIDM**

**git clone -b master https://github.com/ClayBlade/MRIPIDM**

**cd MRIPIDM/MRIPIDM**

nvcc -I/opt/intel/oneapi/ipp/2022.2/include/ipp DoScanAtGPU.cu helperFuncs.cpp -o Bloch -L/opt/intel/oneapi/ipp/latest/lib/intel64 -lipps -lippcore -Xcompiler -fopenmp



Alt: location dependent

sudo apt update \&\& sudo apt upgrade -y

sudo apt install python3 python3-pip python3-venv -y

python3 -m venv venv

**source venv/bin/activate** 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy matplotlib scipy

