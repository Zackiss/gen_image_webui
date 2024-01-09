FROM harbor-intranet.afanticar.com/afanti/python:3.10-groundingdino_swint_ogc

ENV APP="/usr/src/app"

WORKDIR ${APP}

# COPY ./requirements.txt .
COPY . .
#RUN pip3 install --user --upgrade pip

#RUN pip3 install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com  -r requirements.txt
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade -r requirements.txt

