
## Env setup

### Poetry
```
poetry new pytorch-projects
cd pytorch-projects
poetry add torch torchvision matplotlib seaborn 
poetry add fastapi[standard]
poetry add requests rich
```
### UV
```
uv init pytorch-projects
cd pytorch-projects
uv add torch torchvision matplotlib seaborn
uv add fastapi[standard]
uv add requests rich
```