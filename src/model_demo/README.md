




## Environment setup
Poetry environment
```Bash
poetry add torch torchvision matplotlib seaborn 
poetry add fastapi[standard]
poetry add requests
```

To run the scripts and API. First, change the directory to your project `/dev/pytorch-projects`; start a VSCode  editor `code .`; 
you can then generate synthetic data, train a model and test the FastAPI.


```Bash
model_demo/
├── Dockerfile
├── README.md
├── data_prep.py
├── fast_api.py
├── model_demo.py
└── utils.py
```

I run the `data_prep.py` as a script file
```Bash
pytorch-projects-py3.12zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ python  src/model_demo/data_prep.py
# with the following path specification in the scriptimport sys
sys.path.append('/src/model_demo')
from utils import synthesize_data, norm
```
or run as a module
```Bash
pytorch-projects-py3.12zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ python -m src.model_demo.data_prep
# with the alternative specification in the script
from src.model_demo.utils import synthesize_data, norm
```

I configured and run `model_demo.py` as a module with `-m` option. 
```Bash
pytorch-projects-py3.12zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ python -m src.model_demo.model_demo
```
With the following import code (compare to the `data_prep.py` above)
```python
from src.model_demo.utils import LinearRegressionModel, load_data, infer_evaluate_model
```
Note: The command is `python -m src.model_demo.model_demo` not `python -m src.model_demo.model_demo.py`. The later throws error `ModuleNotFoundError: __path__ attribute not found on 'src.model_demo.model_demo' while trying to find 'src.model_demo.model_demo.py'`

To run the server from `pytorch-projects-py3.12zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ `

```Bash
python -m uvicorn src.model_demo.fast_api:app --reload --port 8000
```

To submit test data for inference
```Bash
zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ source .venv/bin/activate
(pytorch-projects-py3.12) zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ python src/model_demo/submit_for_inference.py
```