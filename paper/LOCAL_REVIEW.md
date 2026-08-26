# 本地审阅 LaTeX / PDF

主入口是 `main.tex`，生成的 PDF 固定放在 `build/main.pdf`。

## 最短路径（macOS）

```bash
cd /Users/lukahh2025/Downloads/data_attr_experiment_bundle_20260821/_ICLR_2027__Data_Attribution
make open
```

TinyTeX 已安装在 `~/Library/TinyTeX`，所需 LaTeX 包也已安装。以后只需在该目录运行
`make open`；它会重新编译并用 Preview 打开 PDF。

## 边写边看

如果本机已有完整 TeX Live/MacTeX，可用：

```bash
latexmk -pdf -pvc main.tex
```

VS Code 用户也可以安装 LaTeX Workshop，打开 `main.tex` 后选择 “View LaTeX PDF”。
根文件必须始终设为 `main.tex`，不要单独编译 `sections/*.tex`。

## 当前机器的状态

当前文稿已使用 TinyTeX 2026、`latexmk`、pdfLaTeX 和 BibTeX 完整编译。输出固定为
`build/main.pdf`；编译日志与中间文件也保留在 `build/`，便于定位 warning。
