echo 'Starting Initialization'
pip uninstall -y tensorflow
pip uninstall -y tensorflow-estimator
pip install -r "/content/drive/My Drive/Colab Notebooks/ss19_pa_gesturerecognition_timeconv/requirements.txt"

echo "installing tf_pose and dependencies"

apt-get install swig
apt-get install ffmpeg
cd "/content/drive/My Drive/Colab Notebooks/ss19_pa_gesturerecognition_timeconv/thirdparty/tf-pose/"
git clone https://github.com/ildoonet/tf-pose-estimation.git

cd tf-pose-estimation
pip install -r requirements.txt
cd tf-pose/
rm -y estimator.py
cp ../../../estimator.py .
cd pafprocess/
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

#uncomment if needed
#bash ./models/graph/cmu/download.sh   

#deleting every unneeded file
echo "Cleaning up"
cd "/content/drive/My Drive/Colab Notebooks/ss19_pa_gesturerecognition_timeconv/thirdparty/tf-pose/"
mv tf-pose-estimation/models .
mv tf-pose-estimation/tf_pose .
rm  tf-pose-estimation/*
mv models tf-pose-estimation
mv tf_pose tf-pose-estimation/

echo "successfully installed tf_pose"


echo "installing Pyrealsense2"
apt-get -q install -y libusb-1.0.0-dev
pip install pyrealsense2
echo "successfully installed Pyrealsense2"

echo "successfully installed requirements"
nvcc --version
cat /etc/issue
