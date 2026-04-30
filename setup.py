from setuptools import setup, find_packages

setup(
    name="ecommerce-analytics",
    version="1.0.0",
    description="Indian E-Commerce Analytics  pricing, revenue & demand intelligence",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "ecom-api=api.main:app",
        ]
    },
)
