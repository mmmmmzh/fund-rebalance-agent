# Fund Rebalance Agent

一个离线、可审计的基金研究 Agent 演示项目。它使用 LangGraph 编排逐基金分析、候选研究、风险检查、人工确认、前向验证和运行历史，并通过 Streamlit 提供本地界面。

> 本项目仅用于软件工程、教学和研究，不提供个性化投资建议，不承诺或预测实际收益，不连接交易账户，也不会自动申购、赎回或交易。界面中的“偏强、观察、偏弱、风险退出”均为研究标签，不是交易指令。

## 公开版边界

- 默认只运行确定性合成价格、虚构 ETF 标识和离线规则。
- 支持导入本地持仓 CSV，但只保存白名单字段，不保存账号或身份信息。
- 不包含第三方平台截图 OCR、实时行情抓取、基金报告下载、外部模型调用或 API Key 配置。
- 提供类型化插件接口；插件由使用者自行开发、审查和承担数据授权责任。
- 每日研究项和周五研究池变更在生效前必须人工确认；确认不代表执行真实交易。

## 快速开始

要求 Windows 和 Python 3.11 或更高版本：

```powershell
setup.bat
start.bat
```

本地界面默认打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)。停止服务：

```powershell
stop.ps1
```

手动安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --constraint requirements.lock -e ".[agent,app,dev]"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\fund-agent-dashboard
```

## 本地 CSV

界面接受 UTF-8 或 GB18030 CSV，最大 1 MB、500 行。标准字段：

```csv
code,name,current_weight,current_value_yuan,notes
800001,Demo Broad Market ETF,0.6,,demo
800005,Demo Bond ETF,0.4,,demo
```

也支持对应中文列名。包含账号、用户名、手机号、邮箱、身份证或银行卡等字段的文件会被拒绝；未知列不会保存。真实数据只应放在 Git 忽略的 `user_data/` 中。

## 三类任务

| 任务 | 公开版输入 | 输出 |
| --- | --- | --- |
| 每日研究 | 合成净值、本地组合、内置 Skill | 每只基金的偏强/观察/偏弱/风险退出标签、置信度、风险提示 |
| 周五候选 | 本地或合成候选表 | 排序结果、研究池变更、人工确认记录 |
| 月度主题 | 合成主题序列 | 趋势、波动和方法限制 |

组合目标权重只用于 walk-forward 回测对照，不会转换成交易方向或金额。前向验证账本只记录信号时点和后续方向表现，不记录真实执行。
公开版强制保留人工确认：每日研究项及周五候选研究池变更只有在用户明确批准后才会进入已批准状态；硬风险检查不能被审批绕过。月度主题任务为只读研究，不触发审批。

## 插件接口

[`src/fund_agent/adapters.py`](src/fund_agent/adapters.py) 定义价格、市场上下文、候选、研究证据、Skill 和交易日历协议。公开包不附带任何联网实现。宿主应用必须显式调用 `install_adapters()`，否则 `plugin` 数据模式会直接失败。

## 文档

- [Agent 架构](docs/architecture.md)
- [隐私与本地数据](docs/privacy.md)
- [法律与使用边界](LEGAL.md)
- [数据来源](DATA_SOURCES.md)
- [安全策略](SECURITY.md)
- [第三方说明](THIRD_PARTY_NOTICES.md)

## License

本仓库自行编写的源代码按 [Apache License 2.0](LICENSE) 提供。许可证不授予任何第三方数据、商标、金融产品或插件内容的权利，也不构成对软件适用于投资决策的保证。
