# RS-Toolbox

该项目是针对遥感数据进行切图合并转适量的工具箱, 主要应用于推理时数据的预处理和后处理阶段。

## 1.镜像构建
### 1.1 基础镜像构建
```bash
docker build -f Base.dockerfile -t rs-toolbox:base .
```

### 1.2 运行镜像
```bash
docker build -f Dockerfile -t rs-toolbox:tag .
```

## 2. 镜像使用
### 2.1 启动容器
```bash
docker run -d --name rs-toolbox-1 -p 8080:8080 -v /mnt/tempdata:/mnt/tempdata rs-toolbox:tag
```
访问地址：http://localhost:8080


### 2.2 接口

#### 2.2.1 /api/raster/split_grids 
入参
```bash
image: str = Field(..., description="split raster image file path")
window_size: Optional[tuple] = Field(default=(512, 512), description="window size")
overlap_ratio: Optional[tuple] = Field(default=(0.2, 0.2), description="overlap ratio")
output_path: Union[str, None] = Field(None, description="save path")
    
{
    "image":"user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/gridserver_tif_2024_09_05_18_20_32_949112.tif",
    "window_size": (512, 512)  # Option default(512, 512)
    "overlap_ratio": (0.2, 0.2), # Optional default=(0.2, 0.2)
    output_path: None          # None
}
```
输出
```bash
{
	"code": 0,
	"message": "success",
	"data": {
		"meta_file": "user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/gridserver_tif_2024_09_05_18_20_32_949112/gridinfo.json"
	}
}
```


#### 2.2.2 /api/raster/merge_grids
入参
```bash
meta_file: str = Field(..., description="merge raster image file path")

{
    "meta_file":"user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/gridserver_tif_2024_09_05_18_20_32_949112/gridinfo.json"
}
```
输出
```bash
{
	"code": 0,
	"message": "success",
	"data": {
		"meta_file": "user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/gridserver_tif_2024_09_05_18_20_32_949112/gridinfo.json"
	}
}
```

####  2.2.3 /api/raster/raster_to_vector
入参
```bash
meta_file: str = Field(..., description="merge raster image file path")
specific_label: Union[str, None] = Field(None, description="specific label")
{
    "meta_file":"user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/gridserver_tif_2024_09_05_18_20_32_949112/gridinfo.json"
}

```
输出
```bash
{
	"code": 0,
	"message": "success",
	"data": {
		"meta_file": "user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/gridserver_tif_2024_09_05_18_20_32_949112/gridinfo.json"
	}
}
```




####  2.2.4 /api/raster/det_visualize
描述：用于MMDet模型可视化结果
入参
```bash
meta_file: str = Field(..., description="merge raster image file path")

{
    "meta_file":"user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/gridserver_tif_2024_09_05_18_20_32_949112/gridinfo.json"
}

```
输出
```bash
{
	"code": 0,
	"message": "success",
	"data": {
		"meta_file": "user_temp_data/a09179de-3c62-4ba6-8ed6-75b59e42280a/plane_08/gridinfo.json"
	}
}
```


## 3.更新记录
### 2024-09-xx
rs-toolbox：base 基础版本
### 2024-09-20
    rs-toolbox: v0.0.1 
1. 优化共享盘路径

### 2024-10-21
    rs-toolbox: v0.0.2
1. raster/split_grids接口优化大数据拆分网格逻辑，使用新的slice_bbox方式, 以及响应read write修改
2. raster/merge_grids接口适配slice_bbox数据合并
3. 新增 raster/det_visualize接口， 用于MMDet模型可视化结果

