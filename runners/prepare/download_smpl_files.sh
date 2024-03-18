mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
gdown --fuzzy "https://drive.google.com/file/d/1zHTQ1VrVgr-qGl_ahc0UDgHlXgnwx_lM/view"
rm -rf smpl
rm -rf smplh

unzip smpl.zip
unzip smplh.zip
echo -e "Cleaning\n"
rm smpl.zip
rm smplh.zip

echo -e "Downloading done!"