# 项目文档

## 目录说明

关键目录结构如下：

```
├── _static                                   # css样式文件目录
├── figures                                   # 图片目录
├── python-api                                # Python编程接口文档
│   ├──index.md                               # Python编程接口分类列表
│   ├── ...
├── API_docstring_generation_tool_guide.md    # API文档自动生成工具使用指南
├── Makefile                                  # 自动化脚本文件
├── architecture_introduction.md              # pyasc模块与架构介绍文档
├── conf.py                                   # Sphinx 核心配置文件
├── developer_guide.md                        # Ascend C Python编程接口开发指南
├── index.rst                                 # 文档目录结构定义文件
├── op_debug_prof.md                          # Ascend C Python算子调试调优指南
├── quick_start.md                            # 快速入门文档
└── README
```

## 文档说明

为方便开发者快速熟悉本项目，可按需获取对应文档，文档内容包括：

| 文档                                        | 说明                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| [快速入门文档](quick_start.md)      | 介绍搭建环境、编译执行、本地验证的方式。                               |
| [pyasc模块与架构介绍文档](architecture_introduction.md)    | 介绍pyasc的架构与模块。 |
| [Ascend C Python编程接口开发指南](developer_guide.md)                      | 介绍如何开发Ascend C API的Python编程接口。                                 |
| [Python编程接口列表](python-api/index.md)                      | 项目提供的Ascend C Python编程接口列表。                                 |
| [Ascend C Python算子调试调优指南](op_debug_prof.md) | 介绍常见的算子调试和调优方法。 |
| [pyasc的python语法支持情况说明](python_syntax_support.md) | 介绍pyasc项目支持和不支持的python语法。 |
| [API文档自动生成工具使用指南](API_docstring_generation_tool_guide.md)                 | 介绍自动生成API文档的工具的使用方法。  |
| [PyAsc算子开发指南](pyasc_op_develop_guide.md)                 | 介绍基于pyasc的Ascend C算子开发基本流程。  |

## 附录

| 文档                                | 说明                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Ascend C API参考手册](https://hiascend.com/document/redirect/CannCommunityAscendCApi) | Ascend C所有类别API的使用说明，包括函数原型、使用约束和调用示例等。 |
| [Ascend C算子开发指南](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)   | Ascend C是CANN针对算子开发场景推出的编程语言，使用Ascend C编写算子程序，运行在昇腾AI处理器上，可以实现自定义的创新算法。                       |

