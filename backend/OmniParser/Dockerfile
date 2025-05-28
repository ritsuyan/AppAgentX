# 构建阶段
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel as builder

WORKDIR /app

# 安装构建工具
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 flash-attn
RUN pip install flash-attn --no-build-isolation

# 运行阶段
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装基本的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已编译的 flash-attn
COPY --from=builder /opt/conda/lib/python3.10/site-packages/flash_attn /opt/conda/lib/python3.10/site-packages/flash_attn

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 创建模型和权重文件夹
RUN mkdir -p weights/icon_detect_v1_5 \
    && mkdir -p weights/icon_caption_florence

EXPOSE 8000

CMD ["uvicorn", "omni:app", "--host", "0.0.0.0", "--port", "8000"] 