from pathlib import Path

from setuptools import setup, find_packages


def read_requirements(path: str = "requirements.txt") -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

setup(
    name="ecommerce-analytics",
    version="1.0.0",
    description="Indian E-Commerce Analytics  pricing, revenue & demand intelligence",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "ecom-api=api.main:app",
        ]
    },
)
