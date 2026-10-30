"""Internal BEVFusion deployment implementation; CLI tools live one level above.

Model and dataset adapters, dense/sparse export, TensorRT construction,
numerical validation, and full evaluation belong to separate modules. Package
initialization imports none of them, so engine building and CLI help do not
unconditionally load PyTorch or MMDetection3D.
"""
