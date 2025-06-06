
def safe_torch_load(f, map_location=None, **kwargs):
    """
    安全的torch.load函数，自动处理PyTorch版本兼容性
    
    参数:
    f: 文件路径或文件对象
    map_location: 设备映射，如果为None则使用CPU
    **kwargs: 其他参数
    
    返回:
    torch.load的结果
    """
    import torch
    pytorch_version = torch.__version__.split('.')
    major = int(pytorch_version[0])
    minor = int(pytorch_version[1])
    
    # 如果没有指定map_location，默认使用CPU避免设备不匹配错误
    if map_location is None:
        map_location = 'cpu'
    
    # PyTorch 1.13+ 支持weights_only参数
    if major > 1 or (major == 1 and minor >= 13):
        # 如果没有明确指定weights_only，则设为False以兼容旧的.pt文件
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return torch.load(f, map_location=map_location, **kwargs)
    else:
        # 旧版本PyTorch不支持weights_only参数
        kwargs.pop('weights_only', None)  # 移除weights_only参数
        return torch.load(f, map_location=map_location, **kwargs)
