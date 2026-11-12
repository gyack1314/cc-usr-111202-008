# 分层优先结局的胜比与共享受试者方差

这是研究资料维护用的独立Python项目。`data/`保存合成记录，`research_data.io`负责严格JSON读取与SHA-256来源摘要，`research_data.domain`提供当前原始资料统计，`research_data.cli`将摘要输出为JSON。字段、时间、计数和量纲见[资料约定](docs/data-contract.md)。所有研究、中心及受试者均为虚构。

Python 3.11及以上版本可直接使用，当前没有第三方依赖。在项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q research_data tests
python -m research_data.cli inspect data/normal.json
python -m research_data.cli analyze data/normal.json
python -m research_data.cli analyze data/normal.json --pairs pairs.jsonl --output analysis.json
```

`inspect`只显示已有摘要，不修改来源。`analyze`按层内生存、无进展、生活质量的预先指定优先级做全部跨组配对，输出 W/L、胜比、逐人投影、各结局层的胜负/平局/不可比较数，以及可估时的delta法近似区间。`--pairs`把逐对明细以JSON Lines流式写出，不在内存中保留配对；`--block-size`只改变分块执行方式，不改变数值语义；`--output`把带来源指纹的完整结果写入文件。

方差使用二样本U统计量的一阶投影：对每层治疗行均值向量与对照列均值向量分别求样本协方差（人数减1归一），各自除以该组人数后相加，再按层权重平方合成。逐人投影与分项协方差都在输出中，可逐项重算。每层任一组少于2人时点估计仍列出，但该层及整体的方差和区间留空。W或L为零时保留边界状态，不加入0.5修正。

文件名以`invalid-`起始的资料包含故意不合法的输入，其他文件是同结构的业务边界。数据摘要与来源指纹应同时保留，原始JSON不覆盖。

可使用固定镜像容器：`docker build -t research-material .`，随后`docker run --rm research-material`查看资料；`docker run --rm research-material python -m unittest discover -s tests -v`运行现有检查。全部资料在项目内，运行不需要外部账号或服务。
