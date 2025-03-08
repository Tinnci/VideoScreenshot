# VideoScreenshot

打开视频文件进行按照不同画面自动截图，适用于众多 PPT 讲解视频截取 PPT 图片

## 准备

### 使用传统 pip 安装

```bash
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 推荐：使用 uv 安装（更快、更可靠）

[uv](https://github.com/astral-sh/uv) 是一个极快的 Python 包安装器和解析器，用 Rust 编写。

```bash
# 安装uv
pip install uv

# 使用uv安装依赖
uv pip install opencv-python pillow scikit-image matplotlib requests

# 开发环境还需要安装
uv pip install ruff black
```

## 开发环境设置

本项目使用以下工具进行代码质量控制：

1. **Ruff** - 极快的 Python linter 和 formatter

   - 用于代码检查和格式化
   - 配置在`pyproject.toml`中的`[tool.ruff]`部分

2. **Black** - 无需配置的 Python 代码格式化工具
   - 作为备选格式化工具
   - 配置在`pyproject.toml`中的`[tool.black]`部分

### VSCode 设置

项目包含 VSCode 配置文件，自动设置了以下功能：

- 保存时自动格式化
- 代码检查
- 推荐安装的扩展

### 命令行使用

```bash
# 使用Ruff检查代码
ruff check .

# 使用Ruff格式化代码
ruff format .

# 使用Black格式化代码（备选）
black .
```

## 使用方法

```
usage: vshot.py [-h] [-o OUTDIR] [-S SIMILARITY] [-s SPEND] [-m METHOD] [--all]
                [-q] [-d] [-a ANIMATION] [--dedup-threshold DEDUP_THRESHOLD]
                [--sort] VideoFilePath

用于自动从视频截取不同图片

positional arguments:
  VideoFilePath         输入文件, filepath

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        输出文件夹路径, dirpath
  -S SIMILARITY, --Similarity SIMILARITY
                        相似度参数, 默认低于0.98进行截取, float
  -s SPEND, --spend SPEND
                        间隔帧数, int
  -m METHOD, --method METHOD
                        使用算法, 0,1,2对应均值、差值、感知哈希算法,
                        3,4对应三直方图算法和单通道的直方图,
                        5为ssim(注:该算法效率最低)
  --all                 是否使用让5种算法都进行计算
  -q, --quiet           启用安静模式，减少输出信息
  -d, --dedup           启用自动去重功能，减少相似图片
  -a ANIMATION, --animation ANIMATION
                        动画稳定性检测帧数，默认为0表示不检测，建议值5-15
  --dedup-threshold DEDUP_THRESHOLD
                        去重相似度阈值，默认为0.98
  --sort                根据内容对捕获的图片进行重新排序
```

示例命令:

```bash
# 使用SSIM算法，每帧检查，相似度阈值设为0.999
python vshot.py ./test/ppt_video.mp4 -o ./Testout/ -s 1 -S 0.999 -m 5

# 使用感知哈希算法，每10帧检查，相似度阈值设为0.97
python vshot.py ./test/ppt_video.mp4 -o ./Testout/ -s 10 -S 0.97 -m 2

# 使用安静模式，减少输出信息
python vshot.py ./test/ppt_video.mp4 -o ./Testout/ -s 5 -S 0.98 -m 2 -q

# 针对PPT动画，启用稳定性检测，连续10帧稳定才截图
python vshot.py ./test/ppt_video.mp4 -o ./Testout/ -s 1 -S 0.98 -m 5 -a 10

# 启用捕获后自动去重和排序功能
python vshot.py ./test/ppt_video.mp4 -o ./Testout/ -s 5 -S 0.98 -m 2 -d --sort
```

## 使用建议

1. **算法选择**:

   - 对于 PPT 视频，推荐使用 SSIM 算法(`-m 5`)，虽然效率较低但精度最高
   - 对于变化明显的视频，可以使用感知哈希算法(`-m 2`)提高效率

2. **相似度阈值**:

   - PPT 视频建议使用较高的阈值，如 0.999，以捕获微小变化
   - 普通视频可以使用 0.97-0.98 的阈值

3. **帧间隔**:

   - 对于变化缓慢的视频(如 PPT 讲解)，可以使用较大的间隔(如 10 帧)
   - 对于变化快的视频，推荐使用较小的间隔(如 1-5 帧)

4. **输出模式**:

   - 默认模式下会显示实时进度条和截图信息
   - 对于长视频，可以使用安静模式(`-q`)减少输出信息，只显示进度条和捕获的图片信息

5. **动画稳定性检测**:

   - 对于含有动画效果的 PPT 视频，建议启用动画稳定性检测(`-a`)
   - 稳定性检测参数表示需要连续多少帧稳定才捕获图片，建议值为 5-15
   - 数值越大，稳定性要求越高，越能确保捕获到动画完全结束后的画面

6. **自动去重**:

   - 启用自动去重功能(`-d`)可以删除相似度高于阈值的图片
   - 去重阈值可通过`--dedup-threshold`设置，默认为 0.98
   - 推荐在处理大量视频后使用，减少冗余图片

7. **图片排序**:
   - 启用排序功能(`--sort`)将对捕获的图片进行重新编号
   - 排序后的文件名格式为"sorted_XXX.jpg"
   - 通常与去重功能一起使用，得到有序且无冗余的图片集

## 兼容性信息

本工具已在以下环境中测试通过:

- Python 3.13.1
- OpenCV 4.11.0
- scikit-image 0.25.2
- Pillow 11.1.0
- matplotlib 3.10.1
- requests 2.32.3

如果使用 SSIM 算法(方法 5)遇到问题，可能是因为 scikit-image 的 API 变更，请确保使用正确的导入:

```python
# 新版本导入
from skimage.metrics import structural_similarity
# 旧版本导入(不推荐)
# from skimage.measure import compare_ssim
```

## 其他工具

[DupImageFinder](./other_tools/DupImageFinder/) 用于对于截取的众多图片，进行二次去重。(先点击 reg 注册)

## 常见问题解答

1. **为什么会捕获 PPT 动画过程中的图片？**

   - 默认情况下，程序只会比较相邻帧的差异，无法识别动画过程
   - 使用动画稳定性检测功能(`-a`)可以等待动画完成后再捕获

2. **如何解决捕获的图片太多且有重复？**

   - 使用更高的相似度阈值(`-S`)可以减少捕获
   - 使用自动去重功能(`-d`)可以在捕获后删除相似图片

3. **如何获得最佳的 PPT 截图效果？**
   - 使用 SSIM 算法(`-m 5`)
   - 启用动画稳定性检测(`-a 10`)
   - 使用较小的帧间隔(`-s 1`或`-s 2`)
   - 使用较高的相似度阈值(`-S 0.999`)
   - 处理完成后启用去重和排序(`-d --sort`)
