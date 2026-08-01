---
layout: post
title: 如何让 Codex 使用你的 Kiro 额度
subtitle: 白嫖 Codex 体验的实践记录
tags: [ai, codex, kiro, tools]
comments: true
social-share: true
---

# 如何让 Codex 使用你的 Kiro 额度（本文使用Deepseek优化语言）

> 听朋友说 Codex + GPT 模型有神秘加成，但公司只提供 Kiro 订阅，好在 Kiro 订阅里本身就带了 GPT 模型 —— 这种情况下怎么才能白嫖 Codex 的体验？有的兄弟有的，以下就是我的成功实践。

---

## 前提准备

- **WSL 或其他 Linux 发行版**（本文以 Linux 环境为例）
- **有效的 Kiro 订阅**（个人账号或 SSO 登录均可）
- [kiro-gateway](https://github.com/jwadow/kiro-gateway) —— 把 Kiro 订阅包装成本地 API 网关
- [cc-switch-cli](https://github.com/SaladDay/cc-switch-cli) —— 把 Codex 的 Responses API 请求转换成 Chat Completion API
- **Codex**（终端 AI 助手，安装步骤略）

---

## 第一步：安装并登录 Kiro CLI

Kiro 提供 CLI 和 IDE 两种方式，这里我用 CLI：

1. 根据官方文档下载并安装 `kiro-cli`。
2. 终端执行登录命令，完成账号认证（支持个人账号和 SSO 登录）。

登录成功后，Kiro 会在本地生成认证凭据文件，供后续网关使用。

---

## 第二步：部署 kiro-gateway 本地 API 网关

kiro-gateway 会把你的 Kiro 订阅暴露为一个 **OpenAI 兼容的 Chat Completion API**，但它不能直接对接 Codex，所以后面还需要 cc-switch 做一层转换。

### 2.1 克隆项目并安装依赖

```bash
# 克隆仓库
git clone https://github.com/Jwadow/kiro-gateway.git
cd kiro-gateway

# 安装依赖
pip install -r requirements.txt
```

### 2.2 配置环境变量

```bash
cp .env.example .env
```

用 Vim / nano / 其他编辑器打开 `.env`，修改以下两项：

```ini
# 替换为你的 Kiro SSO 缓存文件路径（通常在 ~/.aws/sso/cache/ 下）
KIRO_CREDS_FILE="~/.aws/sso/cache/你的实际缓存文件.json"

# 为你的本地代理设置一个密码（可自定义，但要记住它）
PROXY_API_KEY="my-super-secret-password-123"
```

> 提示：如果你用的是 AWS SSO 登录，`PROFILE_ARN` 不需要填写，网关会自动处理。

### 2.3 启动网关

```bash
python main.py
```

默认监听 `localhost:8000`，终端会打印出服务启动信息。  
此时你就拥有了一个本地 OpenAI 兼容的 API 端点：`http://localhost:8000/v1`  
**但** Codex 使用的是 Responses API，直接对接会报错，我们需要 cc-switch 来做格式转换。

---

## 第三步：用 cc-switch-cli 搭起 Codex ↔ 网关的桥梁

cc-switch-cli 是一个终端代理工具，能拦截 Codex 发出的 Responses API 请求，并实时转成 Chat Completion API 格式，再转发给 kiro-gateway。

### 3.1 安装 cc-switch-cli 并启动

安装过程请参考官方仓库，完成后在终端运行：

```bash
cc-switch-cli
```

你会进入一个 TUI（终端用户界面）配置环境。

### 3.2 配置 Provider

在 cc-switch-cli 界面中：

1. 按下 `]` 键进入 **Codex 配置页面**。
2. 添加一个新的 Provider，配置如下：
   - **Base URL**：`http://localhost:8000/v1`  
     （如果修改了 kiro-gateway 的端口，请与此保持一致）
   - **API Key**：`my-super-secret-password-123`  
     （就是上一步 `.env` 里设置的 `PROXY_API_KEY`）
   - **Upstream Format**：选择 `chat completions`

配置界面的效果参考下图：

![添加 Provider 配置示例](/assets/img/posts/codex-using-kiro-credits-fig1.png)

### 3.3 模型映射与快速配置

- **Model Mapping**：如果你的 Kiro 订阅本身就包含 GPT 模型，无需手动修改映射；  
  若想在 Codex 里调用其他模型（如 Claude），可以在这里编辑映射关系。
- **Quick Config Menu**：按提示打开快捷配置菜单，**将两个选项全部勾选**（一般包括"启用本地代理"和"转发 API Key"）。
- 按 `Esc` 退出 Quick Config Menu，然后按 `Ctrl + S` 保存所有修改。

### 3.4 启动本地代理

1. 在 cc-switch-cli 中进入 **Settings** 页面，找到 **Local Proxy** 并开启。
2. 按下 `Shift + /` 打开帮助页面，然后按 `T` 键启动本地代理。

代理启动后，cc-switch 就会接管 Codex 的网络请求，自动完成 Responses → Chat Completions 的格式转换。

---

## 第四步：享受你的 Codex

现在整体链路已经打通：

```
Codex 发送 Responses API 请求
        │
        ▼
cc-switch 代理接收请求，转换成 Chat Completion API 格式
        │
        ▼
kiro-gateway 接收 Chat Completion 请求，转发至 Kiro 服务器
        │
        ▼
享受 Codex + GPT 的神秘加成
```

确保三个进程（kiro-gateway、cc-switch-cli 代理、Codex）都在运行，然后你就能愉快地使用 Codex，而所有消耗都走你的 Kiro 额度。

---

