# DeepSeek-R1-Distill-Qwen-7B 模型文件说明
## 分片合并方法
### Linux/Mac
cat model-00001_part_* > model-00001-of-000002.safetensors
cat model-00002_part_* > model-00002-of-000002.safetensors

### Windows
copy /b model-00001_part_aa + model-00001_part_ab + model-00001_part_ac + model-00001_part_ad + model-00001_part_ae model-00001-of-000002.safetensors
copy /b model-00002_part_aa + model-00002_part_ab + model-00002_part_ac + model-00002_part_ad model-00002-of-000002.safetensors
