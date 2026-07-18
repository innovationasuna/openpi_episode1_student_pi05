# enpei pi0.5 适配说明

本仓库 = **最新版 openpi**(原生支持 pi0.5)+ **enpei 适配层**。相比 enpeizhao 原来的 `openpi_episode1_student`(基于 2025-02 的老 openpi),这里是把 enpei 的改动重新搬到带 pi0.5 的最新 openpi 上,并把训练配置切换为 pi0.5。

## 相比官方 openpi 增加/修改的内容

1. **本地数据集 `root` 支持**(来自 enpeizhao 的改动,官方 openpi 没有)
   - `src/openpi/training/config.py`：`DataConfig` 和 `DataConfigFactory` 各加 `root: str | None = None`,并在 `create_base_config` 里透传。
   - `src/openpi/training/data_loader.py`：`create_torch_dataset` 里把 `data_config.root` 传给 `LeRobotDatasetMetadata(repo_id, root=root)` 和 `LeRobotDataset(repo_id, root=root)`。
   - 作用：训练可以直接用本地转换好的数据集(`root="./enpei_dataset/..."`),不必上传 HuggingFace hub。

2. **enpei pi0.5 训练配置**(`src/openpi/training/config.py`)
   - 新增 `enpei_robot_demo_move_fruit_pi05_low_mem_finetune`。
   - 基于官方 `pi05_libero`,改为:`model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, LoRA 变体)`、`data=LeRobotLiberoDataConfig(enpei 数据集 + 本地 root)`、`weight_loader=本地 pi05_base`、LoRA freeze_filter、`ema_decay=None`。
   - **数据适配层(repack / LiberoInputs / LiberoOutputs)与模型无关,pi0 与 pi0.5 通用**,因此直接复用官方 `LeRobotLiberoDataConfig` + `libero_policy.py`。

3. **数据转换脚本**(`examples/libero/lerobot2oppi.py`、`lerobot2oppi_two.py`)
   - 从 enpeizhao 的 fork 原样搬来:把 LeRobot 数据集重打包成 openpi 期望的键(`image`=fixed 相机、`wrist_image`=handeye 相机、`state`、`actions`),并做弧度处理、缩放到 224×224。

4. **依赖钉版**(`pyproject.toml`)
   - `override-dependencies` 增加 `datasets==3.6.0`(enpeizhao 的兼容修复;若与最新 lerobot 冲突可去掉)。

## 使用流程(与原来一致,只是配置名换成 pi05 版)

```bash
# 1. 转换数据(必须弧度制)
uv run ./examples/libero/lerobot2oppi.py \
  --source-repo-id=enpeicv/demo_move_fruit \
  --target-repo-id=enpeicv/demo_move_fruit_openpi \
  --output-path=./enpei_dataset/demo_move_fruit_openpi \
  --source-dataset-root=/root/autodl-tmp/openpi/enpei_dataset/demo_move_fruit \
  --max-episodes=100

# 2. 计算归一化
uv run scripts/compute_norm_stats.py --config-name enpei_robot_demo_move_fruit_pi05_low_mem_finetune

# 3. 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  enpei_robot_demo_move_fruit_pi05_low_mem_finetune --exp-name=my_experiment --overwrite
```

## ⚠️ 必须在 GPU(H20)上验证的点

代码结构已对齐最新 openpi 的真实 API,但以下几处**只有真跑一遍才能确认**:

1. **pi0.5 + LoRA 低显存组合**:官方 pi05 示例都是**全量微调**,没有用 LoRA。`pi05=True` + `gemma_2b_lora` 是否正常收敛/是否报错,需实测。
   - 若有问题,退回**全量微调**:去掉 `paligemma_variant/action_expert_variant/freeze_filter`,改用类似 `pi05_libero` 的全量设置(H20 96G 通常放得下)。
2. **`pi05_base` 权重**:必须下载 pi05_base(pi0_base 用不了),解压到配置里的 `/root/autodl-tmp/pi05_base/params`。
3. **`action_dim` / 状态维度**:enpei 单臂是 7 维;pi0.5 内部按 `action_dim` padding。若报维度错,检查 `Pi0Config.action_dim` 与数据集实际维度。
4. **`datasets==3.6.0` 钉版**:如果 `uv sync` 依赖解析报冲突,先去掉这个 override 再试。
5. **转换脚本的 lerobot 导入路径**:`lerobot2oppi.py` 里用的是 `lerobot.common.datasets...`,若最新 lerobot 改了路径需相应调整。

## 与原 fork 的关系

- 原 fork 的核心模型文件(pi0.py/gemma/tokenizer 等)那些"大改动"其实是**老基准版本差**,不是 enpeizhao 的有意修改,故未搬入——本仓库直接用最新 openpi 的实现。
- 真正搬过来的只有上面 4 类 enpei 功能改动。
