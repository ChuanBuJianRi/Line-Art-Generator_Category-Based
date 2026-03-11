# 分类数据获取与整理说明

每类 **2000 张**，按 70% / 15% / 15% 划分：train 1400、val 300、test 300。三个类别（biological、building、vehicle）分别存到 `dataset_classification/<split>/<类别名>/`。

---

## 1. Biological（动物 + 人，iNaturalist 2021）

**仅保留动物和人**：脚本使用 iNaturalist 的 **kingdom = Animalia**，自动排除植物、真菌等，只抽样动物（含人）图片。

**方式一：用脚本自动下载（推荐）**

脚本会通过 `torchvision.datasets.INaturalist` 下载 2021_train_mini 或 2021_valid，从中只保留 Animalia，再随机抽样 2000 张作为 biological（train 1400、val 300、test 300）。

- 首次运行会下载，需要一定时间和磁盘（约 30GB+）。
- 命令示例见下方「运行脚本」。

**方式二：手动下载后指定路径**

1. 打开 [iNaturalist 2021 竞赛页](https://sites.google.com/view/fgvc8/competitions/inatchallenge2021) 或 [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/i_naturalist2021)。
2. 下载 **train_mini**（如 AWS：`train_mini.tar.gz`），解压到某目录，例如 `~/data/inaturalist2021_mini/`。
3. 运行脚本时加上：`--inaturalist_root ~/data/inaturalist2021_mini`

---

## 2. Building（Places365-Standard）

**方式一：用脚本自动下载（推荐）**

脚本会通过 `torchvision.datasets.Places365` 下载 **small** 版本（256×256，约 30GB），并只保留建筑相关类别（如 apartment_building、church/outdoor、tower、bridge、castle、house、skyscraper、palace、mosque/outdoor、temple/asia 等），再抽样 2000 张（train 1400、val 300、test 300）。

- 命令示例见下方「运行脚本」。

**方式二：手动下载后指定路径**

1. 使用 [places_devkit](https://github.com/zhoubolei/places_devkit) 或 [Places365 官网](http://places2.csail.mit.edu/) 下载 train-standard 的 small 版本。
2. 解压到某目录，例如 `~/data/places365_standard/`。
3. 运行脚本时加上：`--places365_root ~/data/places365_standard`

---

## 3. Vehicle（CompCars 或 Stanford Cars）

Vehicle 需**手动下载**后指定本地目录，脚本会递归查找该目录下所有 `.jpg`/`.png`，随机抽 2000 张作为 vehicle（train 1400、val 300、test 300）。

**方式一：CompCars**

1. 打开 [CompCars 官网](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)，填写 release agreement。
2. 下载 **web-nature 整车图**（full car images），解压到某目录，例如 `~/data/compcars/images/`。
3. 运行脚本时加上：`--compcars_root ~/data/compcars/images`

**方式二：Stanford Cars（可从 Kaggle 下载）**

1. 打开 [Stanford Cars on Kaggle](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars) 或 [Car Connection Picture Dataset](https://www.kaggle.com/datasets/Prondeau/The-Car-Connection-Picture-Dataset) 等，下载并解压到某目录。
2. 运行脚本时加上：`--stanford_cars_root /path/to/car/images`  
   （目录内可含子文件夹，脚本会递归查找图片。）

若暂时没有车辆数据，可只跑 biological 和 building（不加 `--compcars_root` / `--stanford_cars_root`），先得到两类数据。

---

## 4. 运行脚本

在项目根目录执行（需已安装 PyTorch、torchvision）：

```bash
cd "/Users/yiyanggao/Desktop/APS360 Project"
```

**只做 biological + building（自动下载，数据全写外接硬盘）**

用 `--small_download` 时只下较小子集（iNaturalist valid 约 100k 张、Places365 val 约 36k 张），**下载缓存和最终 2000/类 输出都会写在 `--output_dir` 所在盘**（例如外接 SSD），本机几乎不占空间。例如：

```bash
python scripts/prepare_classification_data.py \
  --output_dir "/Volumes/Extreme SSD/APS360_data/dataset_classification" \
  --download_bio \
  --download_building \
  --small_download \
  --seed 42
```

不加 `--small_download` 会下完整 train（体积大），但输出仍是每类 2000 张（train 1400、val 300、test 300）。

**三类都做（车辆数据已下载到指定路径）**

用 CompCars：`--compcars_root /path/to/compcars/images`  
用 Stanford Cars（如 Kaggle）：`--stanford_cars_root /path/to/car/images`

```bash
python scripts/prepare_classification_data.py \
  --output_dir "/Volumes/Extreme SSD/APS360_data/dataset_classification" \
  --download_bio \
  --download_building \
  --compcars_root /path/to/compcars/images \
  --seed 42
```

**全部用已下载数据（不自动下载）**

```bash
python scripts/prepare_classification_data.py \
  --output_dir "/Volumes/Extreme SSD/APS360_data/dataset_classification" \
  --inaturalist_root /path/to/inaturalist2021_mini \
  --places365_root /path/to/places365 \
  --compcars_root /path/to/compcars/images \
  --seed 42
```

- `--output_dir`：输出目录，默认 `dataset_classification`。
- `--seed`：随机种子，保证划分可复现。
- 每类 2000 张，70/15/15 划分：train 1400、val 300、test 300；若某数据源不足 2000 张，脚本会尽量抽满并给出提示。

运行结束后，`dataset_classification/` 下会有 `train/biological`、`train/building`、`train/vehicle` 及对应的 `val/`、`test/`，可直接给 `torchvision.datasets.ImageFolder` 使用。

---

## 5. Store only image data on external SSD (English)

Keep the project/code on your machine; **all dataset data** (download cache + final 2000-per-class output) is written to the external drive when you set `--output_dir` to a path on the SSD.

**1) Find the SSD mount path (Mac)**  
With the SSD connected, run: `ls /Volumes`. Path is `/Volumes/<SSD_name>`. Use quotes if the name has spaces.

**2) Small download + everything on SSD (recommended)**  
Use `--small_download` so only smaller sources are downloaded (iNaturalist valid ~100k images, Places365 val ~36k). Cache and output both go under the same drive as `--output_dir` (e.g. SSD), so the main disk stays free:

```bash
cd "/Users/yiyanggao/Desktop/APS360 Project"

python scripts/prepare_classification_data.py \
  --output_dir "/Volumes/Extreme SSD/APS360_data/dataset_classification" \
  --download_bio \
  --download_building \
  --small_download \
  --seed 42
```

**3) Training: point DataLoader to SSD**  
Use the same path as `root` when loading the dataset:

```python
train_dataset = torchvision.datasets.ImageFolder(
    root="/Volumes/Extreme SSD/APS360_data/dataset_classification/train",
    transform=...
)
```

Only the image data lives on the SSD; code and repo remain on your main drive.
