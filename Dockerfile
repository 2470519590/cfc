FROM python:3.8

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器的 /app 目录
COPY . /app


# 将 requirements.txt 文件复制到 Docker 镜像中
COPY requirement.txt /app/requirement.txt

# 安装依赖
RUN pip install --no-cache-dir -r /app/requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

# 设置容器启动时的默认命令
CMD ["python", "app.py"]
