pip install torch>=1.9.0 torchvision>=0.10.0 numpy>=1.21.1 Pillow>=8.3.1 scikit-learn>=0.24.2 seaborn>=0.11.2 matplotlib>=3.4.2
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2._._+cu___.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2._._+cu___.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2._._+cu___.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2._._+cu___.html
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric

Note: Please replace "torch-2._._+cu___.html" with the version you are using, such as "torch-2.6.0+cu124.html", otherwise the network will encounter errors.
