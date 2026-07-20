# 顶点项目 09 — 代码迁移代理（仓库级语言/运行时升级）

> Amazon的MigrationBench（Java 8到17）和Google的App Engine Py2-to-Py3迁移工具设定了2026年的标准。Moderne的OpenRewrite以确定性方式大规模重写抽象语法树(AST)。Grit通过codemod风格的DSL解决相同问题。生产模式结合了二者：一个用于安全重写的确定性基础层，加上一个处理歧义情况的智能体(Agent)层，一个用于按分支构建的沙箱，以及在PR打开前确保测试通过测试框架。最后的目标是迁移50个真实仓库，并发布通过率及失败分类。

**类型：** 顶点项目
**语言：** Python（智能体），Java / Python（目标），TypeScript（仪表盘）
**前置条件：** 阶段5（NLP），阶段7（Transformer），阶段11（LLM工程），阶段13（工具），阶段14（智能体），阶段15（自主系统），阶段17（基础设施）
**涉及的阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间：** 30小时

## 问题

大规模代码迁移是2026年编码智能体最清晰的生产应用之一。真实情况显而易见（迁移后测试套件是否通过？），回报是真实的（Java 8舰队迁移是一个人力规模项目），基准是公开的（MigrationBench的50仓库子集）。Moderne的OpenRewrite处理确定性部分。智能体层处理OpenRewrite配方无法处理的所有情况：歧义重写、构建系统漂移、长尾语法、传递依赖断裂。

你将构建一个智能体，它接受一个Java 8仓库（或Python 2仓库）并生成一个CI通过的分支。你将测量通过率、测试覆盖率保持情况、每个仓库的成本，并构建一个失败分类。与仅确定性基线的对比，告诉你智能体的价值究竟在哪里。

## 概念

管道有两层。**确定性基础层**（Java用OpenRewrite，Python用libcst）安全地执行大部分机械重写：导入、方法签名、空安全编辑、try-with-resources、废弃API替换。它快速且产生可审计的差异。**智能体层**（OpenAI Agents SDK或基于Claude Opus 4.7和GPT-5.4-Codex的LangGraph）处理配方无法处理的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试不稳定、自定义注解。

每个仓库获得一个Daytona沙箱，其中预安装了目标运行时。智能体迭代：运行构建、分类失败、应用修复、重新运行。硬限制：每个仓库30分钟，每个仓库8美元，20次智能体轮次。如果所有测试通过且覆盖率下降不为负，则分支打开PR。否则，仓库被归类为带有证据的失败类别。

失败分类是可交付成果。在50个仓库中，什么出错了？传递依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类别都有计数和一个示例差异。未来的配方作者可以针对前三名。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性基础层：OpenRewrite（Java）或libcst（Python）
- 智能体：OpenAI Agents SDK或基于Claude Opus 4.7 + GPT-5.4-Codex的LangGraph
- 沙箱：按分支的Daytona devcontainers，预安装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven, Gradle, uv (Python)
- 基准：Amazon MigrationBench的50仓库子集（Java 8到17），Google App Engine Py2-to-Py3仓库
- 测试框架：并行运行器，通过Jacoco（Java）或coverage.py（Python）进行覆盖率
- 可观测性：Langfuse + 每个仓库的跟踪包，包含每个差异块
- 仪表盘：失败分类仪表盘，包含每个类别的计数和示例差异

## 动手构建

1. **配方通过。** 首先运行OpenRewrite（Java）或libcst（Python）配方。捕获70-80%的机械迁移。提交为"recipe"提交。

2. **构建尝试。** Daytona沙箱：安装目标运行时，运行构建。如果通过，跳过到测试。如果失败，交给智能体。

3. **智能体循环。** LangGraph带工具：`run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`。智能体分类失败（依赖、语法、测试、构建工具）并应用针对性修复。重新运行。

4. **预算上限。** 每个仓库30分钟墙钟时间，8美元成本，20次智能体轮次。任何违反则停止并在"budget_exhausted"下归档当前差异。

5. **测试+覆盖率门控。** 构建通过后，运行测试套件。与基础仓库比较覆盖率。如果覆盖率下降超过2%，在"coverage_regression"下归档。

6. **PR打开。** 成功后，推送分支，打开PR，包含差异和摘要，说明应用了哪些配方以及智能体进行了哪些提交。

7. **失败分类。** 对每个失败的仓库，用类别标记：`dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`。构建仪表盘。

8. **50仓库运行。** 在MigrationBench子集上执行。报告每类通过率、每个仓库成本、覆盖率保持情况，以及与仅确定性基线的对比。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 发布

`outputs/skill-migration-agent.md` 是可交付成果。给定一个仓库，它执行确定性配方，然后进行智能体循环以产生一个CI通过的分支，或者将仓库归档到分类类别下。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  MigrationBench通过率  |  50仓库子集 pass@1  |
|  20  |  测试覆盖率保持  |  与基础仓库的平均覆盖率差异  |
|  20  |  每个迁移仓库的成本  |  通过运行的 $/repo  |
|  20  |  智能体/确定性工具集成  |  OpenRewrite处理的修复比例 vs 智能体编写的  |
|  15  |  失败分析报告  |  包含示例的完整分类  |
|  **100**  |   |   |

## 练习

1. 仅使用OpenRewrite（无智能体）运行迁移管道。与完整管道比较通过率。识别智能体独自发挥作用的案例。

2. 实现"lint-clean"检查：迁移后，运行风格检查器（Java用spotless，Python用ruff）。如果出现新的lint错误，则PR失败。测量覆盖率保持但风格退化的比率。

3. 添加"最小差异"优化器：智能体的分支通过测试后，通过第二轮修剪不必要的更改。报告差异大小减少。

4. 扩展到第三个迁移：Node 18到Node 22。重用沙箱包装；将配方层替换为自定义codemod。

5. 测量首次构建通过时间(TTFGB)作为用户体验指标。目标：p50在10分钟以下。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  确定性基础层  |  "配方引擎"  |  OpenRewrite / libcst：具有安全保证的声明式AST重写  |
|  Codemod  |  "代码修改程序"  |  机械地更改源代码的重写规则  |
| 构建漂移 | "工具版本偏差" | 主要版本之间 Maven / Gradle / uv 行为的细微变化 |
| 失败类别 | "分类桶" | 仓库未迁移的标记原因：依赖(dep)、语法(syntax)、测试(test)、构建工具(build-tool)、预算(budget) |
| 覆盖率增量 | "覆盖率保持" | 从基准分支到迁移分支的测试覆盖率百分比变化 |
| 智能体回合 | "工具调用轮次" | 智能体循环中的一个计划->行动->观察周期 |
| 预算耗尽 | "触及上限" | 仓库消耗了其30分钟/8美元/20回合的限制而未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 规范的2026基准测试
- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 确定性基底参考
- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 配方编写
- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 替代代码修改DSL
- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — Agents SDK参考
- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 替代迁移基准测试
- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — Python确定性基底
- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 参考每分支沙箱
