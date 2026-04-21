# 人脸检测与相似度检索 API

## 功能说明

- 左侧页面：上传图片后进行人脸检测，返回人脸框、置信度、裁剪后的人脸。
- 右侧页面：上传图片后自动入库，按文件名去重，再基于人脸特征做 Top-K 相似度检索。
- 数据库存储：使用 SQLite，默认数据库文件在 `face_web_api/runtime_data/face_images.db`。
- 图片存储：默认存储到 `face_web_api/runtime_data/uploads/`。

## 安装依赖

```bash
pip install -r requirements-face-api.txt
```

## 启动方式

```bash
python run_face_api.py
```

或者：

```bash
uvicorn face_web_api.app:app --host 0.0.0.0 --port 8000 --reload
```

## 打开页面

浏览器访问：

```text
http://127.0.0.1:8000
```

## 权重说明

- 默认情况下，代码会尝试使用 `facenet_pytorch` 的 `vggface2` 预训练权重。
- 如果你已经下载了本地权重，可以设置环境变量 `FACE_RECOG_WEIGHTS` 指向权重路径。

示例：

```bash
export FACE_RECOG_WEIGHTS=/your/path/20180402-114759-vggface2.pt
python run_face_api.py
```
