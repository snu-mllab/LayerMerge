from setuptools import setup, find_packages

setup(
    name="layer_merge",
    version="0.1",
    packages=find_packages(),
    package_data={
        "layer_merge": [
            "kim23efficient/*.txt",
            "kim24layermerge/*.txt",
            "models/ddpm_cfg/*.yml",
        ]
    },
    entry_points={
        "console_scripts": [
            "lymg_kim23_dp = layer_merge.kim23efficient.generate_tables:main",
            "lymg_kim23_imp = layer_merge.kim23efficient.importance:main",
            "lymg_kim24_dp = layer_merge.kim24layermerge.generate_tables:main",
            "lymg_kim24_imp = layer_merge.kim24layermerge.importance:main",
            "lymg_kim24lyr_dp = layer_merge.kim24layer.generate_tables:main",
            "lymg_kim24lyr_imp = layer_merge.kim24layer.importance:main",
            "lymg_agg = layer_merge.aggregate_imp:main",
        ]
    },
)
