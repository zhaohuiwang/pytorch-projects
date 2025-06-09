
## Env setup
```
uv init --python 3.12
uv add torch torchvision matplotlib seaborn 
```
Encounter an error message `ModuleNotFoundError: No module named '_ctypes'`
this error just wont go away when using poetry instead 
```
poetry add torch torchvision matplotlib seaborn 
poetry add fastapi[standard]
```