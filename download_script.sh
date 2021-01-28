# this is used for drive directly downloads
function download-google(){
  echo "https://drive.google.com/uc?export=download&id=$1"
  mkdir -p .tmp
  curl -c .tmp/$1cookies "https://drive.google.com/uc?export=download&id=$1" > .tmp/$1intermezzo.html;
  curl -L -b .tmp/$1cookies "$(egrep -o "https.+download" .tmp/$1intermezzo.html)" > $2;
}

# some files are shared using an indirect download
function download-google-2(){
  echo "https://drive.google.com/uc?export=download&id=$1"
  mkdir -p .tmp
  curl -c .tmp/$1cookies "https://drive.google.com/uc?export=download&id=$1" > .tmp/$1intermezzo.html;
  code=$(egrep -o "confirm=(.+)&amp;id=" .tmp/$1intermezzo.html | cut -d"=" -f2 | cut -d"&" -f1)
  curl -L -b .tmp/$1cookies "https://drive.google.com/uc?export=download&confirm=$code&id=$1" > $2;
}
declare -f
# used like this
#download-google <id> <name of item.extension>
download-google-2 '1elEebJuMn8Ey372UPh-OgqWDdiHpi-yD' phoneCall.tar.gz