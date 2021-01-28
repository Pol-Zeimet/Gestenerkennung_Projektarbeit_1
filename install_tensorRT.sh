cd "/content/drive/My Drive/Colab Notebooks/ss19_pa_gesturerecognition_timeconv/"
dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.0-trt7.0.0.11-ga-20191216_1-1_amd64.deb
apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda10.0-trt7.0.0.11-ga-20191216/7fa2af80.pub
apt-get update
apt-get install tensorrt
apt-get install python3-libnvinfer-dev
apt-get install uff-converter-tf
echo "Install complete. Validating install: "
dpkg -l | grep TensorRT