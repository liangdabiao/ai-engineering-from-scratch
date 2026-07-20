# 生产环境中的MCP认证 — 注册、JWKS刷新、受众固定令牌

> 第16课在内存中搭建了OAuth 2.1状态机。到2026年，你交付给真实组织的每个MCP服务器都将位于生产认证之后：可扩展至无限客户端群体的客户端注册（首先使用客户端ID元数据文档，动态客户端注册作为向后兼容的备用方案）、授权服务器元数据发现（RFC 8414 *或* OpenID Connect发现）、不破坏凌晨3点令牌验证的JWKS缓存刷新，以及拒绝跨资源重放的受众固定令牌。本节课使用三个角色（授权服务器、资源服务器（MCP服务器）和客户端）对整个流程建模，使你能够追踪从发现到已验证工具调用的每一步。
>
> **规范说明（2025-11-25）：** 2025年11月的MCP授权规范将动态客户端注册从`SHOULD`降级为`MAY`，并将**客户端ID元数据文档（CIMD）**设为推荐的默认注册机制。本节课按照规范的优先级顺序讲授两者，代码保留了DCR以进行演示，因为它完全自包含在一个进程中。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 阶段13·16（OAuth 2.1状态机），阶段13·17（网关）
**时间：** 约90分钟

## 学习目标

- 通过RFC 8414元数据发现授权服务器并验证合同。
- 实现RFC 7591动态客户端注册，使MCP客户端无需管理干预即可注册。
- 按计划缓存和刷新JWKS密钥，使签名验证在密钥轮换后仍能正常工作。
- 使用RFC 8707资源指示符将令牌固定到单个MCP资源，并拒绝混淆代理重用。
- 清晰分离三个角色——授权服务器、资源服务器、客户端——每个只执行属于自己的检查。
- 读取IdP能力矩阵，当IdP无法满足MCP的认证配置时拒绝部署。

## 问题

第16课的模拟器在内存中运行OAuth 2.1。生产环境存在三个操作缺口，纯内存模拟器无法发现。

第一个缺口是注册。真实组织运行着数百个MCP服务器和数千个MCP客户端。操作员不会手动将每个Cursor用户注册为OAuth客户端。2025-11-25规范为客户端提供了解决问题的优先级顺序：如果有预注册的`client_id`，则使用它；否则使用**客户端ID元数据文档**（客户端使用其控制的HTTPS URL标识自己，授权服务器*拉取*元数据）；否则回退到**RFC 7591动态客户端注册**（客户端*推送*一个`POST /register`并立即收到一个`client_id`）；否则提示用户。CIMD是推荐的默认方式，因为它完全消除了每服务器注册，同时保持基于DNS的信任模型；DCR为向后兼容性而保留。两者都从授权服务器的元数据中发现其入口点：`client_id_metadata_document_supported`用于CIMD，`registration_endpoint`用于DCR。

第二个缺口是密钥轮换。JWT验证依赖于授权服务器的签名密钥，这些密钥以JSON Web密钥集（JWKS）的形式发布。授权服务器会按计划轮换这些密钥（通常每小时一次，在事件响应时可能更快）。MCP服务器在启动时只获取一次JWKS，在轮换窗口之前验证正常——然后每个请求都失败，直到重启。生产环境将JWKS作为缓存值，并设置一个刷新作业在旧密钥过期前覆盖缓存，同时当缓存未命中时（当用比缓存更新的密钥签名的令牌到达时）进行回退获取。

第三个缺口是受众绑定。第16课引入了RFC 8707资源指示符。在生产环境中，该指示符成为每个请求的硬性声明检查。MCP服务器将`token.aud`与其自身的规范资源URL进行比较，并拒绝不匹配的请求（HTTP 401）。这是防止上游MCP服务器（或持有针对某服务器令牌的恶意客户端）在相同信任网格中重放该令牌到另一服务器的唯一防御。

本节课将每个缺口映射到流程的具体部分。元数据文档是一个HTTP端点。JWKS缓存刷新是一个定时作业加上键值缓存。JWT验证是资源服务器在调度任何工具之前运行的例程。保持三个角色分离，每个只执行属于自己的检查：授权服务器签发和轮换密钥，资源服务器缓存和验证，客户端发现和注册。

## 核心概念

### RFC 8414 — OAuth授权服务器元数据

位于`/.well-known/oauth-authorization-server`的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定MCP资源URL的客户端会链式发现：来自RFC 9728（资源服务器的文档）的`oauth-protected-resource`命名颁发者，然后来自`oauth-authorization-server`（本RFC）命名每个端点。客户端永不硬编码授权URL。

在信任某个IdP用于MCP之前需验证的合同：

- `code_challenge_methods_supported`包括`S256`（基于RFC 7636的PKCE）。规范明确说明：如果该字段**缺失**，则授权服务器不支持PKCE，客户端**必须**拒绝继续。
- `code_challenge_methods_supported`包括`S256`并拒绝`grant_types_supported`和`authorization_code`。
- 至少有一个注册路径被公布：`code_challenge_methods_supported`（CIMD，首选）**或**`S256`（RFC 7591 DCR，回退）。任一都满足合同；不再硬性要求DCR。
- `code_challenge_methods_supported`恰好是用于OAuth 2.1的`S256`。

如果`S256`缺失，则MCP服务器拒绝针对此IdP部署——PKCE没有降级模式。如果两个注册路径都未公布且你没有预注册的`client_id`，你也无法注册；部署清单出了问题，而不是代码。

### RFC 9728（回顾）——受保护资源元数据

第16课涵盖了RFC 9728。生产环境中的差异：此文档是客户端查找*此*MCP服务器信任的授权服务器的唯一位置。单个MCP服务器可以接受来自多个IdP的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728声明了该集合；RFC 8414记录了每个IdP支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### 客户端ID元数据文档（推荐的默认方式）

CIMD将注册从*推送*反转为*拉取*。客户端不是要求授权服务器创建一个`client_id`，而是使用其控制的HTTPS URL作为`client_id`。该URL解析为一个JSON元数据文档；授权服务器在OAuth流程期间按需获取它。信任植根于DNS：如果服务器操作员信任`app.example.com`，则信任来自`https://app.example.com/client.json`的客户端。无需注册往返，无需`client_id`命名空间耗尽，无需保持每服务器状态同步。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的`client_id`值**必须**等于其服务地址的URL（授权服务器验证此点；不匹配则被拒绝）。授权服务器通过其RFC 8414元数据中的`client_id_metadata_document_supported: true`公布支持。

规范中明确指出的两个安全事实：

- **SSRF。** 授权服务器获取攻击者提供的URL。它必须防御服务器端请求伪造（不获取内部/管理端点）。
- **localhost冒充。** CIMD本身无法阻止本地攻击者声明合法客户端的元数据URL并绑定任何`localhost`重定向。授权服务器**必须**在同意期间清晰显示重定向URI主机名，并且**应该**在仅限`localhost`的重定向时发出警告。

由于CIMD不需要服务器端状态，因此无需像DCR所需那样建立一个注册中心。客户端侧是只读的：从静态HTTPS端点提供你的元数据文档，让授权服务器拉取它。

### RFC 7591 — 动态客户端注册（回退/向后兼容）

DCR现在是一个`MAY`，为与2025-11-25之前的部署以及尚未支持CIMD的IdP向后兼容而保留。没有它（以及没有CIMD或预注册），每个MCP客户端（Cursor、Claude Desktop、自定义代理）都需要与IdP管理员进行带外交互。使用DCR，客户端发布：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应一个`client_id`和一个用于后续更新的`registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none`是运行在用户设备上的MCP客户端的正确默认值。它们只获得一个`client_id`——没有可供泄露的`client_secret`。PKCE提供了公共客户端所需的持有证明。

三个生产陷阱：

- 注册端点必须按源IP进行速率限制。否则，恶意行为者通过脚本进行数百万虚假注册，耗尽`client_id`命名空间。在注册中心处理请求之前执行速率限制检查。
- 某些企业IdP要求`client_id`（一个为客户端担保的签名JWT）。本课的模拟跳过了它；生产环境需要添加验证步骤，拒绝来自非localhost重定向URI的未签名注册。
- `client_id`必须存储为哈希值，而不是明文。该令牌被盗意味着攻击者可以重写客户端的重定向URI。

### RFC 8707 (回顾) — 资源指示器

第16课确立了这一模式。生产规则：每次令牌请求都包含`resource=<canonical-mcp-url>`，MCP服务器在每次调用时验证`token.aud`是否与其自身的资源URL匹配。规范URI是服务器的最具体标识符：它使用小写协议和主机，没有片段，并且通常没有尾部斜杠。路径组件**不**被规则去除——规范在需要标识单个MCP服务器时会保留路径。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443`和`https://mcp.example.com/server/mcp`都是有效的规范URI。为每个服务器选择一个，并将`aud`精确锁定为该URI。（本课的模拟为了简洁，使用了裸主机受众如`https://notes.example.com`；在同一源下托管多个MCP服务器的部署通过路径区分它们。）

### RFC 7636 (回顾) — PKCE

PKCE在OAuth 2.1中是强制性的。本课的授权码流程始终携带`code_challenge`和`code_verifier`。服务器拒绝任何没有验证器或验证器的哈希值与存储的挑战不匹配的令牌请求。

### MCP规范2025-11-25认证配置文件

MCP规范(2025-11-25)精确规定了MCP服务器的授权层必须做什么：

- 实现RFC 9728受保护资源元数据，并通过`WWW-Authenticate: Bearer resource_metadata="..."`头在401响应中或其知名URI `/.well-known/oauth-protected-resource`(SEP-985使头成为可选的，并提供了知名URI回退)提供其位置。元数据`authorization_servers`字段**必须**命名至少一个服务器。
- 仅通过`WWW-Authenticate: Bearer resource_metadata="..."`在**每次**请求中接受令牌——绝不在查询字符串中，也绝不在会话开始时仅验证一次。
- 在每个请求中验证`WWW-Authenticate: Bearer resource_metadata="..."`、`/.well-known/oauth-protected-resource`、`authorization_servers`和所需的作用域。服务器**必须**验证令牌是专门为其颁发的（受众）；缺失或不匹配的`Authorization: Bearer ...`将被拒绝，绝不视为通配符。
- 在401/403响应中，返回携带`WWW-Authenticate: Bearer resource_metadata="..."`的`/.well-known/oauth-protected-resource`，`authorization_servers`参数（元数据文档的URL，*不是*裸资源），以及`Authorization: Bearer ...`在`aud`(403)上。注意：参数是`iss`，一个发现指针——挑战中没有`exp`参数。
- 授权服务器发现接受**要么**RFC 8414 OAuth元数据**要么**OpenID Connect Discovery 1.0；客户端必须按优先级顺序尝试两个知名后缀。
- 客户端（而非服务器）防御**混合攻击**：它在重定向前记录预期的`WWW-Authenticate: Bearer resource_metadata="..."`，并在兑换代码前验证`/.well-known/oauth-protected-resource`授权响应参数(RFC 9207)。仅PKCE不能阻止混合攻击，因为客户端将其`authorization_servers`交给了它被引导到的任何令牌端点。

OAuth 2.1草案是基础；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD是表层；MCP规范是配置文件。

### IdP能力矩阵

并非所有IdP都支持完整的MCP配置文件。下面的矩阵记录了截至2025-11-25规范的事实能力声明。它是一个*部署关口*，而不是建议。

CIMD随2025-11-25规范发布，底层OAuth草案仅在2025年10月被采纳，因此供应商支持仍在陆续推出——将下面的"CIMD"视为"当前状态，在你的租户中验证"，而不是永久声明。

|  IdP类别  |  AS元数据(8414/OIDC)  |  CIMD  |  RFC 7591 DCR  |  RFC 8707资源  |  RFC 7636 S256 PKCE  |  备注  |
|---|---|---|---|---|---|---|
|  自托管(Keycloak)  |  是  |  新兴  |  是  |  是(自24.x起)  |  是  |  本课MCP配置文件的参考IdP；完整的端到端DCR路径，CIMD跟踪新规范。  |
|  企业SSO(Microsoft Entra ID)  |  是  |  新兴  |  是(高级层级)  |  是  |  是  |  DCR可用性因租户层级而异；部署前在目标租户中验证。  |
|  企业SSO(Okta)  |  是  |  新兴  |  是(Okta CIC / Auth0)  |  是  |  是  |  DCR在Auth0(现Okta CIC)上可用；经典Okta组织需要管理员预注册。  |
|  社交登录IdP(通用)  |  因情况而异  |  否  |  极少  |  极少  |  是  |  大多数社交IdP将客户端视为静态合作伙伴；无自助注册。仅用作身份源，在上层构建你自己的MCP感知授权服务器。  |
|  自定义/自建  |  取决于  |  取决于  |  取决于  |  取决于  |  取决于  |  如果你自行发布，请发布完整配置文件并优先使用CIMD。跳过PKCE或受众绑定会破坏MCP认证契约。  |

部署清单的拒绝规则：如果所选IdP在`code_challenge_methods_supported`中没有列出`S256`，MCP服务器拒绝启动——PKCE没有降级模式。注册是一个较软的关口：你需要*一个*可用的路径（预注册的`client_id`、`client_id_metadata_document_supported: true`或`registration_endpoint`）。仅缺少DCR不再是拒绝触发器，因为CIMD或预注册可以覆盖它。

### JWKS刷新模式（在AS轮换，在资源服务器刷新）

将两个动词分开，因为混用它们是一个真实的生产环境错误：

- **轮换**是*授权服务器*做的事情：生成一个新的签名密钥，在JWKS中发布，稍后淘汰旧的。资源服务器不参与此事，也无法做到——它不持有IdP的私钥。
- **刷新**是*资源服务器*做的事情：将已发布的JWKS重新`GET`到其缓存中。这是资源服务器唯一执行的JWKS操作。

生产环境的故障模式是缓存过期。通过定时刷新作业和键值缓存来解决。资源服务器运行一个作业（cron、定时器，运行时提供的任何机制），在固定间隔内获取`<issuer>/.well-known/jwks.json`并覆盖`cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取。如果令牌的`kid`在缓存中缺失，则触发**一次**同步刷新作为回退，然后重新检查。这同时处理了两种情况：定时刷新，以及在新密钥签名的令牌在下一个定时刷新之前到达时的密钥重叠窗口。

回退**必须是重新获取，绝不能是轮换**。如果将缓存未命中路径连接到轮换和生成，两件事会出问题：(1) 生成新密钥会产生一个*仍然*不匹配令牌的`kid`，因此查找仍然失败；以及 (2) 攻击者用随机`kid`值喷射令牌会导致无限次密钥创建——自我造成的DoS。重新获取是幂等的，因此虚假的`kid`最多浪费一次获取。

缓存形状：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时有两个密钥是稳定状态。授权服务器通过在淘汰前一个密钥(`k_2026_03`)之前引入下一个密钥(`k_2026_04`)进行轮换，因此旧密钥下签发的令牌在到期前仍然有效。缓存持有并集；验证器根据`kid`选择。

### 验证程序

MCP服务器在调度任何工具之前运行验证。`code/main.py`使用的形状：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate`解码JWT，从JWKS缓存中解析签名密钥（未命中则刷新一次），验证签名，然后检查`iss`是否在白名单中，`aud`是否与此服务器的规范资源匹配，`exp`，以及所需的作用域——在第一次失败时返回`WWW-Authenticate`挑战。将其作为资源服务器上的一个单一例程意味着每个入口点（每个工具调用，每个传输）都通过相同的检查；没有不先验证就到达工具的路径。

### 受众重放演练（访问令牌权限限制）

服务器A(`notes.example.com`)和服务器B(`tasks.example.com`)都在同一个授权服务器上注册。服务器A被攻破。攻击者获取用户的笔记令牌并针对服务器B进行重放。

服务器B的验证器：

1. 解码JWT，通过`kid`获取JWKS，验证签名。
2. 检查`kid`是否匹配其受保护资源元数据中的`iss`。（通过——相同IdP。）
3. 检查`kid`。（失败——令牌的`iss`是`authorization_servers`。）
4. 返回401并附带`kid`。

受众声明是协议层对此攻击的唯一防御。为了性能而跳过它是生产环境中最常见的错误；验证器必须在每个请求上运行，而不仅仅在会话开始时。规范将此称为**访问令牌权限限制**：一个MCP服务器`MUST`拒绝任何未在受众中指定它的令牌。

> **命名说明。** 规范将术语*confused deputy*保留给一个相关但不同的问题：MCP服务器作为OAuth**代理**访问第三方API，使用静态客户端ID，转发令牌而未获得每个客户端用户同意。受众绑定修复了上述重放攻击；confused deputy的修复是每个客户端同意**加上**永不将入站令牌传递给上游API（MCP服务器`MUST`获取自己的独立上游令牌）。

### 混合攻击（服务器无法提供的客户端防御）

一个客户端在其生命周期中与多个授权服务器通信。恶意AS可以尝试让客户端在攻击者的令牌端点兑换诚实AS的授权码。受众绑定在此处无帮助——攻击发生在任何令牌存在之前。防御位于客户端（RFC 9207）：

1. 在重定向之前，客户端从验证后的AS元数据中记录预期的`issuer`。
2. 在授权响应中，客户端将返回的`issuer`参数与记录的issuer进行比较（简单字符串比较，不进行规范化），然后再将代码发送到任何地方。
3. 不匹配（或当AS声明`iss`时`issuer`缺失）→拒绝，甚至不显示`iss`字段。

PKCE单独无法阻止混合攻击，因为客户端将其`code_verifier`交给任何被引导到的令牌端点。这就是规范在每个请求中记录issuer以及PKCE验证器和`state`的原因。

### 故障模式

- **过期的JWKS。** 验证器在AS轮换密钥后拒绝有效令牌。修复方法是上述定期刷新+缓存未命中重新获取模式。永远不要在没有刷新任务的情况下缓存JWKS。
- **轮换作为回退。** 将缓存未命中路径连接到轮换并签发而不是重新获取是一个真正的错误：它永远不会产生缺失的`kid`，并将攻击者控制的`kid`值变成密钥创建DoS。回退必须是幂等的`refresh-jwks`。
- **缺失`kid`声明。** 一些IdP默认省略`kid`，除非令牌请求中存在`refresh-jwks`。验证器必须拒绝缺失`aud`的令牌，而不是将缺失视为通配符。
- **通过缺失`kid`检查的混合攻击。** 如果客户端未验证RFC 9207的`kid`授权响应参数与重定向前记录的issuer是否匹配，则可能被引导到在攻击者的令牌端点兑换诚实AS的代码。这是客户端侧失败；资源服务器无法补偿。
- **权限升级竞争。** 同一用户的两个并发升级流程可能都成功，并产生两个具有不同作用域的访问令牌。验证器必须使用请求中呈现的令牌，而不是查找“用户的当前作用域”——这会产生TOCTOU窗口。
- **注册令牌泄露。** 泄露的`kid`允许攻击者重写重定向URI。对这些内容进行哈希存储；要求客户端在每次更新时提供明文；怀疑泄露时轮换。
- **`kid`未固定。** 接受任何`kid`的验证器会允许攻击者建立自己的授权服务器，为目标受众注册客户端，并签发令牌。受保护资源元数据的`refresh-jwks`列表是允许列表；强制执行它。

## 使用它

`code/main.py`使用stdlib Python和三个角色——`AuthorizationServer`、`ResourceServer`和`Client`——演示完整的生产流程。流程如下：

1. 授权服务器在`/.well-known/oauth-authorization-server`发布RFC 8414元数据。
2. MCP客户端调用元数据端点并检查其注册选项（CIMD的`/.well-known/oauth-authorization-server`，DCR的`client_id_metadata_document_supported`）以及`registration_endpoint` PKCE支持。
3. 演练采用DCR回退路径：客户端向`/.well-known/oauth-authorization-server`（RFC 7591）发送请求并收到`client_id_metadata_document_supported`。（CIMD客户端将呈现自己的HTTPS `registration_endpoint` URL并跳过此步骤。）
4. MCP客户端运行PKCE保护的授权码流程（RFC 7636），附带`/.well-known/oauth-authorization-server`指示符（RFC 8707）。
5. MCP客户端使用`/.well-known/oauth-authorization-server`调用MCP服务器上的工具。
6. MCP服务器运行`/.well-known/oauth-authorization-server`，从JWKS缓存中解析签名密钥。
7. IdP轮换一个密钥；计划刷新重新拉取JWKS到缓存中。
8. 下一次调用使用刷新后的密钥进行验证，无需重启，且之前的令牌在重叠窗口期间仍然有效。
9. 针对不同MCP资源的受众重放尝试会收到401，附带`/.well-known/oauth-authorization-server`和`client_id_metadata_document_supported`指针。

此处的JWT使用HS256和共享密钥（因此课程仅使用stdlib运行）。生产环境使用RS256或EdDSA配合上述JWKS模式；验证逻辑在其他方面相同。由于IdP和资源服务器在同一进程中，`refresh_jwks`直接读取授权服务器的密钥列表；通过线路传输时是HTTP `GET`到`jwks_uri`。

## 发布

本课程产出`outputs/skill-mcp-auth.md`。给定MCP服务器配置和IdP能力集，技能输出需要建立的认证表面——受保护资源元数据、使用的注册路径（CIMD、预注册或DCR回退）、JWKS刷新计划、作用域映射，以及当IdP不支持完整RFC配置时应用的拒绝规则。

## 练习

1. 运行`code/main.py`。追踪流程。注意IdP如何在步骤6中轮换密钥，计划`refresh_jwks`重新拉取已发布的集合，以及旧令牌（重叠窗口）和新令牌无需重启即可验证。

2. 向受保护资源元数据的`authorization_servers`列表添加一个新的IdP。签发一个由新IdP签名的令牌，确认验证器接受它。签发一个由未列出的IdP签名的令牌，确认验证器拒绝并返回`WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 在注册器接受请求之前，向`register_client`添加速率限制检查。使用每个源IP的令牌桶，保存在由IP键控的小型字典中。

4. 阅读RFC 7591，识别本课程`/register`处理程序未验证的两个字段。添加验证。（提示：`software_statement`和`redirect_uris` URI方案。）

5. 添加一个客户端ID元数据文档路径。提供一个`client.json`，其`client_id`等于它自己的URL，并让授权服务器获取并验证它（如果`client_id` ≠ URL则拒绝）。确认CIMD客户端无需`register_client`调用即可注册。

6. 证明DoS修复。向验证器发送一个带有随机`kid`的令牌，确认`refresh_jwks`最多运行一次，且授权服务器的密钥计数不增长。然后故意将回退重新连接到轮换并签发，观察每个伪造令牌的密钥计数增长——之后恢复重新获取。

7. 实现客户端侧RFC 9207 `iss`检查（来自混合攻击部分）：在授权请求前记录预期的issuer，然后拒绝`iss`不匹配的授权响应。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  ASM  |  "OAuth元数据文档"  |  RFC 8414 `/.well-known/oauth-authorization-server` JSON  |
|  CIMD  |  "客户端元数据URL"  |  客户端ID元数据文档——作为`client_id`的HTTPS URL；AS拉取该JSON。自2025-11-25起的推荐默认值  |
|  DCR  |  "自助客户端注册"  |  RFC 7591 `POST /register`流程；在2025-11-25中降级为`MAY`回退  |
|  JWKS  |  "用于JWT验证的公钥"  |  JSON Web Key Set，从`jwks_uri`获取，通过`kid`索引  |
|  轮换 vs 刷新  |  "更新密钥"  |  *轮换* = AS创建/废弃签名密钥；*刷新* = 资源服务器重新获取已发布的集合。资源服务器只能刷新  |
|  资源指示符  |  "受众参数"  |  RFC 8707 `resource`参数，将令牌固定到一个服务器  |
|  `aud`声明  |  "受众"  |  JWT声明，验证器将其与规范资源URL进行比较  |
|  受众重放  |  "令牌重放"  |  为服务器A签发的令牌被呈现给服务器B；通过受众验证防御（规范：访问令牌权限限制）  |
|  Confused deputy  |  "代理令牌滥用"  |  具有静态客户端ID的MCP代理转发令牌而未获得每个客户端同意；与受众重放不同  |
|  混合攻击  |  "错误令牌端点"  |  客户端被引导到在攻击者的端点兑换诚实AS的代码；通过RFC 9207 `iss`在客户端侧防御  |
|  `iss` 允许列表  |  "受信任的授权服务器"  |  受保护资源元数据的 `authorization_servers` 中命名的集合 |
|  `resource_metadata`  |  "在哪里找到 PRM 文档"  |  在 401/403 上命名 RFC 9728 元数据 URL 的 `WWW-Authenticate` 参数 |
|  公开客户端  |  "原生或浏览器客户端"  |  没有 `client_secret` 的 OAuth 客户端；PKCE 进行补偿 |
|  `WWW-Authenticate`  |  "401/403 响应头"  |  携带驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课程实现的 MCP 认证配置文件
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 2025-11-25 的变化（CIMD、XAA、DCR 降级）
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — CIMD-over-DCR 的原理
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — CIMD
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 发现合同
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — DCR（回退路径）
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 公开客户端持有证明
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 受众锁定
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 资源服务器发现
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 防御混淆攻击的 [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) 参数
- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 统一的 OAuth 基础
