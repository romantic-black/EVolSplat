from types import ModuleType
import data_info


def load_data_info(module_name, data_info_dict=None, mldb_type="mldb_info", module=None, _visited=None):
    """
    Recursively collect `mldb_info` dicts from the given data_info module/package.

    Args:
        module_name (str): Name of the top-level module, usually "data_info".
        data_info_dict (dict): Dict to be populated with collected info.
        mldb_type (str): Key name to look for inside modules (default: "mldb_info").
        module (ModuleType | None): Internal use – current module being traversed.
    """
    if data_info_dict is None:
        data_info_dict = {}

    if _visited is None:
        _visited = set()

    # Resolve the starting module only once from globals()
    if module is None:
        module = globals().get(module_name, None)

    if not module:
        raise RuntimeError(f'Try to access "{mldb_type}", but cannot find {module_name} module.')

    # Avoid infinite recursion across cyclic module graphs
    if module in _visited:
        return data_info_dict
    _visited.add(module)

    for key, value in module.__dict__.items():
        if key.startswith("__") or key.startswith("_"):
            continue

        if key == mldb_type:
            # Merge found mldb_info dict into the accumulator
            data_info_dict.update(value)
        elif isinstance(value, ModuleType):
            load_data_info(
                module_name + "." + key,
                data_info_dict,
                mldb_type=mldb_type,
                module=value,
                _visited=_visited,
            )

    return data_info_dict

def reset_ckpt_path(cfg, data_info):
    """
    Recursively update checkpoint paths in config.
    Supports both dict and Config objects (mmcv/mmengine Config inherits from dict).
    """
    if isinstance(cfg, dict):
        for key in cfg.keys():
            if key == 'backbone':
                backbone = cfg.get('backbone')
                if backbone is not None and isinstance(backbone, dict):
                    backbone_type = backbone.get('type')
                    if backbone_type and backbone_type in data_info.get('checkpoint', {}):
                        new_ckpt_path = data_info['checkpoint']['mldb_root'] + '/' + data_info['checkpoint'][backbone_type]
                        backbone['checkpoint'] = new_ckpt_path
                        # Config 对象也支持 update 方法，确保同步更新
                        if hasattr(backbone, 'update'):
                            backbone.update(checkpoint=new_ckpt_path)
                continue
            elif isinstance(cfg.get(key), dict):
                reset_ckpt_path(cfg.get(key), data_info)
            else:
                continue
    else:
        return

if __name__ == "__main__":
    mldb_info_tmp = {}
    load_data_info("data_info", mldb_info_tmp)
    print("results", mldb_info_tmp.keys())