FROM python:3.8

# 设置工作目录
WORKDIR /

# 复制当前目录内容到容器的 /app 目录
COPY . /app

# 安装依赖
RUN pip install --no-cache-dir -r requirement.txt

# 设置容器启动时的默认命令
CMD ["python", "app.py"]