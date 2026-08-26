# 本地审阅 LaTeX / PDF

主入口是 `main.tex`，生成的 PDF 固定放在 `build/main.pdf`。

## 最短路径（macOS）

```bash
cd data_attribution/paper
HOMEBREW_NO_AUTO_UPDATE=1 brew install tectonic
make open
```

以后只需在该目录运行 `make open`；它会重新编译并用 Preview 打开 PDF。

## 边写边看

如果本机已有完整 TeX Live/MacTeX，可用：

```bash
latexmk -pdf -pvc main.tex
```

VS Code 用户也可以安装 LaTeX Workshop，打开 `main.tex` 后选择 “View LaTeX PDF”。
根文件必须始终设为 `main.tex`，不要单独编译 `sections/*.tex`。

## 当前机器的状态

当前环境尚未安装 `tectonic`、`latexmk` 或 `pdflatex`，因此本次只完成了源码静态检查，
没有提交由未验证工具链生成的 PDF。安装上述任一工具链后，`make open` 即可生成并审阅。
