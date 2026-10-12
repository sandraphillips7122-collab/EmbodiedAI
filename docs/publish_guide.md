# 发布指引（把本仓库推到 GitHub）

> 目的：让**面试官能直接点开查看** —— 目前仓库只在本机，这是对求职影响最大的一个短板，且补齐成本最低。

## 1. 推送前检查清单

### 1.1 确认敏感/体积大的内容未被跟踪

```cmd
cd /d <你的仓库根目录>
:: 看受控文件里有没有不该入库的东西
git ls-files | findstr /i "\.venv \.cache experiments third_party .pt .h5 .mp4"
:: 应为空；若列出内容，说明 .gitignore 没覆盖到
```

本仓库的 `.gitignore` 已覆盖：`.venv/`、`experiments/`、`*.h5`、`*.pt`、`*.ckpt`、`*.safetensors`、`*.mp4`、`.cache/`、`.maniskill/`。
**受控文件应为 60 个左右**（可用 `git ls-files | wc -l` 核对，取决于是否已提交本次新增文件）。

### 1.2 确认没有泄露本机隐私

```cmd
:: 检查是否把绝对路径 / 用户名硬编码进了受控文件
git grep -n -i "C:/Users" -- "*.py" "*.md" "*.toml" "*.txt"
```

**已知情况**：`reports/` 与 `docs/` 里出现过 `C:\Users\<user>...` 这类**示例路径**（用于说明文件位置），
属于可接受的文档示例；但**必须确认没有出现你的真实用户名**。若有，改成 `<user>` 再提交。

### 1.3 确认历史里没有大文件（可选但建议）

```cmd
:: 若仓库体积异常（>30 MB），说明历史里混入过大文件
git count-objects -vH
```

## 2. 创建远端并推送

```cmd
:: 方式 A：用 GitHub CLI（若已安装 gh）
gh repo create EmbodiedAI --public --source=. --remote=origin --push

:: 方式 B：先在网页建空仓库，再手动关联
git remote add origin https://github.com/<你的用户名>/EmbodiedAI.git
git branch -M main
git push -u origin main
git push origin v1.0        :: 把 tag 也推上去
```

> **建议仓库名**：`EmbodiedAI` 或 `maniskill-act-and-smolvla`（后者更直白地说明内容）。

## 3. README 首屏建议（面试官 30 秒内看到什么）

当前 `README.md` 开头已是「项目状态表 + 主结果」，**已经很适合**。若想进一步优化，建议：

1. **第一屏就放结果表**（现状已满足）—— 不要用大段背景介绍开头
2. 在状态表下面加一行**一句话定位**，例如：
   > 两个项目共享同一评测协议，**A 为完整正结果闭环，B 为方法论严谨的负结果诊断**；所有数字均可由独立代码复算。
3. **加一个"如何验证结论"的醒目入口**（现状已有 `## 如何验证这些结论` 一节）——
   这是本仓库与其他学生项目最不同的地方，值得放在显眼位置
4. 可选：录一个 GIF（`scripts/demo_gradio.py` 的界面）放在首屏 —— 但注意**不要**把 Demo 当成主要成果，
   它只是"可视化入口"；主成果是实验与评测体系

## 4. 推上去之后的建议动作

- **给仓库加 topics**：`robotics`、`imitation-learning`、`vla`、`manipulation`、`maniskill`、`lerobot`
- **Pin 到个人主页**（GitHub Profile → Customize pins）
- 若某份报告想单独分享，可直接把 `reports/A.md` / `reports/B.md` 的链接发给对方

## 5. 不建议做的事

- ❌ **不要**为了"好看"把项目 B 的负结果删掉或改成正结果 —— 那会毁掉这个仓库最有价值的部分（严谨性）
- ❌ **不要**把 `experiments/` 里的 JSONL/MP4 传上去（体积大且无必要；文档里已说明如何重跑）
- ❌ **不要**把 `.venv/` 或本地 wheel 缓存传上去
