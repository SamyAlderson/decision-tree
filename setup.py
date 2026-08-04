from setuptools import setup, find_packages

try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    print("Warning: README.md not found. Skipping long description.")
except Exception as e:
    print(f"Error reading README.md: {e}")

# Remove dead code
# Removed print statements as they're not necessary

setup(
    name='decision-tree',
    version='1.0',
    description='A basic decision tree implementation in Python for data science tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/samy-alderson/decision-tree',
    author='Samy Alderson',
    author_email='samy.alderson@example.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy', 'pandas'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.9',
)