# 第三方说明

本项目依赖 Python 开源包，包括 pandas、NumPy、SciPy、Matplotlib、Pydantic、LangGraph、Streamlit 及其传递依赖。具体锁定版本见 `requirements.lock`，各包版权与许可证以其发行包元数据和上游仓库为准。

当前锁文件中的直接运行依赖：

| 包 | 锁定版本 | 上游许可证标识 |
| --- | --- | --- |
| Matplotlib | 3.11.0 | Python Software Foundation License（以发行包许可证文件为准） |
| NumPy | 2.4.6 | BSD-3-Clause 及发行包声明的组件许可证 |
| pandas | 3.0.3 | BSD License |
| Pillow | 12.3.0 | MIT-CMU |
| Pydantic | 2.13.4 | MIT |
| SciPy | 1.17.1 | BSD License |
| LangGraph | 1.2.9 | MIT |
| LangGraph SQLite checkpoint | 3.1.0 | MIT |
| Streamlit | 1.59.2 | Apache-2.0 |

该表由安装包元数据整理，不替代各上游项目随发行包提供的完整许可证与 NOTICE。传递依赖的锁定版本见 `requirements.lock`。

Apache License 2.0 只覆盖本仓库自行编写并有权许可的内容，不替代第三方依赖的许可证。本项目不随仓库再分发第三方实时行情、基金报告、模型权重、平台截图或账户数据。

GitHub Actions 在 CI 中按固定提交使用，其使用仍受对应上游许可证和 GitHub 服务条款约束。
