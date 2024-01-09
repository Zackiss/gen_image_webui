```shell
docker start gen
docker exec -it gen bash
conda activate lora
cd /gen-image-webui
python main.py
git config --global credential.helper store
git config credential.helper store
nohup ./v2ray -c ./config.json 
docker cp gen:/gen-image-webui/outputs .
docker cp gen:/gen-image-webui/test .
```

