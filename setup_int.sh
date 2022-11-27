mkdir data
mkdir data/int
mkdir data/raw
wget -O data/int/applications.zip https://storage.googleapis.com/cs229-sat-embeddings/applications-20221027T231202Z-001.zip
unzip data/int/applications.zip -d data/int/
rm data/int/applications.zip

wget -O data/int/CONTUS_UAR.pkl https://storage.googleapis.com/cs229-sat-embeddings/CONTUS_UAR.pkl

mkdir data/raw/eval_images
wget -O data/raw/eval_images/images.zip https://storage.googleapis.com/cs229-sat-embeddings/new_mtl_images.zip
unzip data/raw/eval_images/images.zip -d data/raw/eval_images
rm data/raw/eval_images/images.zip
mv data/raw/eval_images/content/mosaiks_images/* data/raw/eval_images
rm -rf data/raw/eval_images/content