




## Environment setup
Poetry environment
```Bash
poetry add torch torchvision matplotlib seaborn 
poetry add fastapi[standard]
poetry add requests rich
```

To run the scripts and API. First, change the directory to your project `/dev/pytorch-projects`; start a VSCode  editor `code .`; you may need to activate the envoronment: `source .venv/bin/activate` after which you should expect `(pytorch-projects-py3.12)` at the begining of the bash/szh prompt indicating you are inside a Python environment. 
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

If you like to execute the `data_prep.py` as a script file, follow this instruction
```Bash
pytorch-projects-py3.12zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ python  src/model_demo/data_prep.py
# with the following path specification in the scriptimport sys
sys.path.append('/src/model_demo')
from utils import synthesize_data, norm
```
or if you want to run it as a module
```Bash
pytorch-projects-py3.12zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ python -m src.model_demo.data_prep
# with the alternative specification in the script
from src.model_demo.utils import synthesize_data, norm
```

I only configured and run `model_demo.py` as a module with `-m` option. 
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
To submit test data for inference through web URL http://localhost:8000/docs by Swagger UI. (typical localhost IP address is 127.0.0.1, so alternatively you may through http://127.0.0.1:8000/docs instead. Run `cat /etc/hosts` from terminal to confirm the IP address) 

Go to Post > [Try it out] > input data into "Request body" box > [Execute]

To submit test data for inference through Python script
```Bash
zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ source .venv/bin/activate
(pytorch-projects-py3.12) zhaohuiwang@WangFamily:/mnt/e/zhaohuiwang/dev/pytorch-projects$ python src/model_demo/submit_for_inference.py
```

To see the prediction result from `http://localhost:8000/docs`
Alternatively, use curl
`curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{ "feature_X_1": 4,"feature_X_2": 7}'`