echo -e "Downloading T2M evaluators"
gdown --fuzzy https://drive.google.com/file/d/1ZL81tHLaGA3D7ZhLcbc7JKEs40OgzLov/view
gdown --fuzzy https://drive.google.com/file/d/1nNZOSlYxDjyuUHAXzauSWsEFgRi0N5ON/view
rm -rf t2m

unzip t2m.zip
unzip dataset.zip
echo -e "Cleaning\n"
rm t2m.zip
rm dataset.zip

echo -e "Downloading done!"