# FlightCast：Windows 前端开发环境

## 当前版本（2026-10-20）

网页界面、左右分屏历史记录、日期与时间选择已完成。认证与历史记录新增 40 项测试全部通过，原有 6 项后端测试及前端构建也通过，详见 [测试说明](AUTH_HISTORY_TEST_REPORT.md)。日常启动仍使用下文入口；本次账号更新后需要重启后端并重新登录一次，历史记录保留。

下面包含本机首次配置时的实际路径。其他组员应将项目目录、Python 与 Node 路径替换为自己的安装位置；也可以直接使用已安装的 Python 3.12 和 Node 24。仓库保留原始模型 metadata，本机绝对路径仅在运行 `scripts/prepare_local.py --apply` 后适配，不作为团队模型修改提交。

测试与开发产生的 `.local/`、`.venv/`、`node_modules/`、`dist/`、原始 CSV、账号数据库和登录密钥不上传 GitHub。已在仓库中的现役模型文件保持原样。

## 首次环境配置记录（2026-10-18）

**环境已配置完成，并已通过前后端联调。** Python 独立环境、前端依赖、原始数据、本机模型路径和 SQLite 元数据库均已准备好。后端原有 6 项测试全部通过，前端构建通过，浏览器页面 HTTP 200，前端代理下的真实预测成功。四个模型组件（`cls / single / bucket+iso / cond`）全部参与，天气来源为 `Open-Meteo forecast`，五档概率之和为 1。

日常使用：双击根目录里的 `启动后端.cmd` 与 `启动前端.cmd`，保持两个窗口运行，再打开 http://127.0.0.1:5173。按 Ctrl+C 或关闭对应窗口停止；如果页面已在运行，就不必再次启动。下文一次性配置步骤保留作环境重建参考，目前无需重复安装。

实际项目根目录是两层同名文件夹中的内层：

```text
D:\desktop\26-27 semA\IS6400\group project\IS6400-main\IS6400-main
```

本次新增环境说明、Python 网页依赖清单及完整锁文件、数据准备脚本、启动入口和忽略规则。14 个模型的 metadata 仅修改 `model_path` 本机路径，并保留原件备份；已用 SHA-256 核验模型二进制未变，同时与原 ZIP 对比确认原前后端源码、模型注册表及前端依赖锁文件未变。README 中的训练、数据集评估、Git 提交等步骤属于项目文档，本次没有执行。

## 1. 项目需要什么

| 部分 | 本项目配置 | 用途 |
| --- | --- | --- |
| 前端 | Vue 3.5.42、Vite 8.3.3（来自 package-lock.json） | 表单、预测结果、登录与历史记录 |
| Node | 建议 Node 24 LTS；现有普通命令是 20.16.0，不满足要求 | 安装前端依赖、运行开发服务器 |
| Python | 建议 3.12；模型记录为 3.12.14 | Flask 与模型推理 |
| 模型库 | CatBoost 1.2.10、LightGBM 4.7.0（来自模型 metadata） | 当前模型和启动时先执行的基线预测 |
| 数据库 | Python 自带 SQLite | 机场/航司/距离查询，以及用户历史 |
| 天气 | 后端调用 Open-Meteo，代码没有要求 API key | 起飞、到达两端天气特征 |

本地开发不需要 GPU、CUDA、MySQL、Docker，也不需要重新训练或下载历史天气。

Vite 和 Vue 插件在锁文件中都要求 Node `^20.19.0 || >=22.12.0`，也可参考 [Vite 官方说明](https://vite.dev/guide/)。建议长期独立开发时安装 [Node 24 LTS](https://nodejs.org/en/download) 和 [Python 3.12](https://www.python.org/downloads/windows/)。下面先复用电脑上已有的可用运行时，不修改全局环境变量。

## 2. 一次性配置

在 PowerShell 中按顺序执行。每一段没有报错再执行下一段。

### 2.1 进入正确目录

```powershell
Set-Location -LiteralPath 'D:\desktop\26-27 semA\IS6400\group project\IS6400-main\IS6400-main'
```

### 2.2 创建独立 Python 环境，安装网页依赖

已确认以下本机解释器存在，版本为 Python 3.12.14：

```powershell
$projectBasePython = 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $projectBasePython -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\requirements-web.lock.txt
& .\.venv\Scripts\python.exe -m pip check
```

这里直接调用虚拟环境的解释器，无须运行 Activate.ps1。虚拟环境仍依赖上面这个基础 Python 的位置；将来如果该运行时被删除，需要用自己安装的 Python 3.12 重新创建 `.venv`，不能把 `.venv` 当成可搬运的完整 Python。参见 [Python venv 文档](https://docs.python.org/3.12/library/venv.html)。

`requirements-web.txt` 记录直接依赖；`requirements-web.lock.txt` 固定了本机实际安装并验证通过的全部 26 个依赖，包括 CatBoost 1.2.10、LightGBM 4.7.0、pandas 3.0.6、numpy 2.5.3、Flask 3.1.3。重建环境优先使用完整锁文件。如果出现 `No matching distribution found`，先保留错误信息，检查 Python 版本与当前包源是否提供该版本，不要为消除报错而改模型文件或盲目降级。

### 2.3 准备数据库和模型的本机路径

初次检查发现的问题及本次处理结果：

- 当前四个天气模型与十个基线/区间模型文件，共 14 个，均存在；概率校准 JSON 也存在。
- 模型 metadata 的 `model_path` 原本指向组长电脑的 `/Users/...` 路径。基线加载器直接使用这个路径；本次已经完成适配并备份原件。
- `Dataset` 中原本没有原始 CSV，`server/flight_meta.db` 也未构建；本次均已准备好，数据库包含 18 家航司、380 个机场、7,785 条航线。
- 旧目录中的 `archive\flights_sample_3m.csv` 已核验 MD5，和项目清单一致：`633c02070a85e801b6bf23d76acd3fdf`。文件约 586 MiB。

先只检查（不改文件）：

```powershell
& .\.venv\Scripts\python.exe .\scripts\prepare_local.py --source-csv 'D:\desktop\26-27 semA\IS6400\group project\archive\flights_sample_3m.csv'
```

然后应用本机配置：

```powershell
& .\.venv\Scripts\python.exe .\scripts\prepare_local.py --apply --source-csv 'D:\desktop\26-27 semA\IS6400\group project\archive\flights_sample_3m.csv'
```

脚本会校验并复制 CSV 到新项目的 Dataset；备份 14 个模型的原始 metadata 到 `.local/metadata-backup/`，只替换 `model_path` 字段；记录并复核模型权重的 SHA-256；数据库不存在时运行原有 `build_meta_db.py`。不会改模型注册表、特征、模型权重或推理代码，也不会覆盖已有数据库。

这是本地环境适配；这些绝对路径不要作为团队模型更新提交。备份目录保留了原始 metadata，后续可逐个恢复。换电脑应重新准备本机路径。

**仅运行网页不需要执行 `01_prepare_data.py`**：检查代码可见，元数据库构建直接读取原始 CSV；预测读取已有模型。清洗 CSV 是训练/评估流程的需要。

### 2.4 安装前端依赖

电脑上另有 Node 24.19.0 可以使用。通过它执行现有 npm，避免误用系统 Node 20.16.0：

```powershell
$projectNode = 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$projectNpmCli = 'D:\Program Files\Node.js\node_modules\npm\bin\npm-cli.js'
$env:PATH = (Split-Path -Parent $projectNode) + ';' + $env:PATH
Push-Location .\flight_delay_model\frontend
& $projectNode $projectNpmCli ci
Pop-Location
```

这里使用 `npm ci` 按现有 `package-lock.json` 安装，不修改团队锁文件；如果已有 node_modules，它会重新创建依赖目录。参见 [npm ci 官方说明](https://docs.npmjs.com/cli/v11/commands/npm-ci/)。这一段的 PATH 调整只作用于当前终端。

## 3. 日常开发：两个终端

两个终端都先进入上面的实际项目根目录。

终端 A 启动 Flask：

```powershell
.\dev.ps1 -Target backend
```

终端 B 启动前端：

```powershell
.\dev.ps1 -Target frontend
```

浏览器访问 **http://127.0.0.1:5173**。保存 Vue/CSS 后，开发服务器会自动更新页面。Flask 地址是 **http://127.0.0.1:5000**，现有 vite.config.js 已将 `/api` 代理到这里，无须新增跨域配置，也无须改现有接口。

如果 PowerShell 提示禁止执行脚本，可以仅对这一次启动使用以下命令，不必修改系统执行策略：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\dev.ps1 -Target backend
```

另一个终端把末尾改为 `-Target frontend`。按 Ctrl+C 停止对应服务。如果端口被占用，先关闭已运行的该项目进程；前端脚本固定使用 5173，占用时会明确报错。

`dev.ps1` 优先使用 PATH 中满足版本要求的 Node，否则使用已检查到的本机 Node 24；仅临时调整当前脚本环境。Python 始终使用本项目 `.venv`。

## 4. 安装完成后的验收

```powershell
.\dev.ps1 -Target check
.\dev.ps1 -Target build
.\dev.ps1 -Target weather
```

- `check`：运行原有 6 项后端测试，预期 `6 passed, 0 failed`；包括真实模型预测、搜索、距离补全、登录与历史存储逻辑。它不是训练或重跑 2023 数据集评估。
- `build`：前端能生成 dist。随后仅启动 Flask，访问 http://127.0.0.1:5000，也能查看构建后的页面。
- `weather`：单独验证实时天气网络与两端特征。后端测试使用旧日期，不能单靠后端测试通过来证明实时预报已接通。天气自检失败时需单独处理网络问题；原服务已有天气缺失与基线回退逻辑。

前端开发服务器运行时还可以检查代理和接口：

```powershell
Invoke-RestMethod 'http://127.0.0.1:5173/api/airlines?q=DL'
Invoke-RestMethod 'http://127.0.0.1:5173/api/airports?q=ATL'
```

最后在网页选择近期日期、Delta、ATL → DEN，填写出发/到达当地时间（现有输入是 HHMM，例如 `1830`、`2015`），距离留空，点击预测。应看到五档概率、结论和延误分钟数；检查响应中的 `weather_components` 是否包含 `cls`、`single`、`bucket+iso`、`cond`，确认当前四组件已参与。实时天气还需查看 `weather_source`，不能把缺失天气视为预报成功。

## 5. 下一步界面优化的范围

环境验收之后，主要编辑：

| 文件 | 负责内容 |
| --- | --- |
| `flight_delay_model/frontend/src/App.vue` | 页面布局、航班表单、结果区、历史页 |
| `flight_delay_model/frontend/src/style.css` | 全局色彩、字体、间距、响应式布局 |
| `flight_delay_model/frontend/src/components/SearchSelect.vue` | 航司/机场搜索下拉 |
| `flight_delay_model/frontend/src/components/DelayPanel.vue` | 五档概率和结果展示 |
| `flight_delay_model/frontend/src/components/AuthModal.vue` | 登录注册弹窗 |

保留 `/api/airlines`、`/api/airports`、`/api/predict`、`/api/auth/*`、`/api/history*` 的调用与字段。输入仍为 `FL_DATE / AIRLINE / ORIGIN / DEST / CRS_DEP_TIME / CRS_ARR_TIME`，`DISTANCE` 可选。

当前模型由 `models/wx_serving.json` 指向的四个 CatBoost 组件组成：延误概率、五档概率加校准、延误条件下分钟数、单段式分钟预测。界面优化保持这些模型选择和返回含义。尤其 `expected_minutes_overall` 是单段式中位口径，可能为 0；它与五档加权分钟数 `expected_delay_min_from_buckets` 不是同一个指标，设计时应分别标注，避免混用。

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| Python 命令打开商店或找不到 Python | 使用上述解释器完整路径，或安装自己的 Python 3.12 |
| 找不到 pip | `.venv` 创建尚未成功，重新运行 2.2 的 venv 命令并处理其错误 |
| Node 版本警告、Vite 启动失败 | 避免 Node 20.16.0，确认实际使用 Node 24 |
| SQLite 提示 `no such table: airlines` | 尚未成功生成 flight_meta.db，完成 2.3；空数据库文件不能代替元数据 |
| 模型报 `/Users/...` 文件不存在 | 尚未应用 metadata 本机路径适配，完成 2.3 |
| 页面可打开，下拉搜索失败或代理拒绝连接 | 确认 Flask 在 5000 运行，且浏览器访问的是 5173 |
| Flask 的 5000 首页显示仅后端提示 | 尚未 build；开发时访问 5173，或先构建前端 |
| 实时天气不可用 | 检查网络和所选日期；旧日期或超出预报窗口不代表模型缺失 |

登录数据库 `server/app.db` 和会话密钥 `server/.secret_key` 会由原后端生成；无需手工编造。依赖、备份和本地数据库已经由忽略规则覆盖。
