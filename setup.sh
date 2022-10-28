mkdir data
mkdir data/int
mkdir data/raw
wget -O data/int/applications.zip https://storage.googleapis.com/cs229-sat-embeddings/applications-20221027T231202Z-001.zip
unzip data/int/applications.zip -d data/int/
rm data/int/applications.zip

wget -O data/raw/images.zip https://storage.googleapis.com/cs229-sat-embeddings/images.zip
unzip data/raw/images.zip -d data/raw/
rm data/raw/images.zip
