import yaml


def load_config(path, default_path=None):
    """
    加载配置文件，并支持“继承”机制。

    Args:
        path (str): 目标配置文件路径
        default_path (str, optional): 默认配置文件路径（兜底用）
    
    Returns:
        dict: 合并后的配置字典
    """

    # 1. 先读取当前配置文件
    with open(path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # 2. 检查当前配置里是否定义了 "inherit_from"
    #    - 如果有：说明需要先加载父配置
    #    - 如果没有：则看是否提供了 default_path
    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        # 递归调用，先加载父配置作为基础
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        # 如果没有继承，但调用时提供了 default_path，就加载它
        with open(default_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        # 如果既没有继承，也没有 default_path，就从空字典开始
        cfg = dict()

    # 3. 用当前配置覆盖父配置/默认配置
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    递归地合并两个配置字典。
    - dict1: 原始配置（父配置 / 默认配置）
    - dict2: 新配置（当前配置）

    合并规则：
    1. 如果 dict2 的 value 是字典，递归合并。
    2. 如果 dict2 的 value 是普通值，直接覆盖 dict1 的值。
    3. 冲突时，dict2（子配置）的优先级更高。
    """

    for k, v in dict2.items():
        if isinstance(v, dict):
            # 如果 key 在 dict1 里不存在，先补一个空字典
            if k not in dict1:
                dict1[k] = dict()
            # 递归继续更新子字典
            update_recursive(dict1[k], v)
        else:
            # 如果是普通值，直接覆盖
            dict1[k] = v
