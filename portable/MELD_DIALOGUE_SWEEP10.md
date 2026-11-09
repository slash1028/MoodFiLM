# 单卡 MELD 对话人格 MoodFiLM：10 settings

参考：89_smf_prompt_upper6_0902/meld_pp_b050 第4 epoch，test WF1=70.232，ACC=71.188。
这是已核实的简化 full 参考，不宣称整个项目所有架构的最高分。
从同一 Qwen 基础权重重新训练，不从见过测试集选模的最终 ERC adapter 继续训练。

保持旧配置的 w6、压缩音频 tokens（audio_context_window=6）、LR=1e-4、
lambda=.20、gamma=.45、beta=.50、prior=.01、dynamics=.0005；每组5 epochs。
这里 w6 沿用原 builder：最多6句（包含目标句），不是6句历史加目标句。
单卡 microbatch1、累积40、有效batch40。单卡50 epochs不保证一晚完成。

|编号|人格来源/变化|seed|
|---|---|---|
|01|PELD，完整MoodFiLM+prompt|42|
|02|对话预测，完整MoodFiLM+prompt|42|
|03|PELD，完整MoodFiLM+prompt|43|
|04|对话预测，完整MoodFiLM+prompt|43|
|05|仅训练集预测聚合的角色均值，完整MoodFiLM+prompt|42|
|06|对话预测，MoodFiLM，无人格prompt|42|
|07|对话预测，仅人格prompt，无MoodFiLM及辅助loss|42|
|08|无人格prompt、无MoodFiLM及辅助loss，保留文本+音频|42|
|09|对话预测full，LR=7e-5|42|
|10|对话预测full，lambda=.30|42|

只用 split/dialogue_id/speaker 精确匹配，OCEAN保持1–7原始尺度，不人为扩散预测。
PELD、对话预测和训练角色均值采用同一可用性掩码；缺失三条和非主要角色未知，
不回退到PELD或填均值。不会删除ERC句子，test仍2610条。
因此PELD是当前代码/单卡/覆盖率匹配的参考重跑，不保证等于历史70.232。
新人格使用全对话，可能含ERC目标之后的句子；结果必须标为离线全对话协议，
不可声称严格因果ERC。这批实验不验证人格GT准确率。

## 运行（yiwang账号，在tmux内，确认GPU0可用）

    cd /n/work6/yiwang/MoodFilm/portable && GPU=0 EPOCHS=5 PYTHON_BIN=/n/work3/huang/tools/miniconda3/envs/InstructERC_cu12/bin/python bash run_meld_dialogue10_1gpu.sh --run

不加 --run 只准备数据和校验命令，不启动训练。
默认选择此目录下最新的完整 export/dialogue_personality.csv（不会选smoke_export）。
可 --prediction-csv /明确路径/dialogue_personality.csv 指定来源。
预测未完整导出/存在未允许的缺失或重复时立即停止。不会等待或伪造标签。

输出：meld_dialogue_sweeps/meld_dialogue10_1027/。
每个setting实时终端输出，attempts/*.log保存日志，adapter_epoch*保存adapter。
epoch_metrics.csv记录各epoch WF1和ACC；best_test_metrics.json是探索性test最佳值，
不是独立验证集选模结果。coverage.json记录缺失、匹配范围和协议限制。

重复同一命令跳过 DONE 的设置；中断且已有训练产物的setting不会覆盖，
请用新的 --run-id 显式开新实验。输入/代码/超参变动也要求新run-id。
节点本地/tmp缓存用于Triton/编译；最终实验产物都在项目子目录。

准备阶段仅做CPU/静态与合成数据测试；未做真实训练/GPU显存验证。
